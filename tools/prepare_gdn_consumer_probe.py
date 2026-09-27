#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

def main():
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('--tokens',type=int,default=16); p.add_argument('--heads',type=int,default=16); p.add_argument('--key-dim',type=int,default=128); p.add_argument('--value-dim',type=int,default=128); p.add_argument('--hidden',type=int,default=1024); p.add_argument('--model-dir'); p.add_argument('--layer',type=int,default=17); a=p.parse_args()
    rng=np.random.default_rng(20260919); sh=(a.tokens,a.heads)
    q=rng.normal(0,.2,(*sh,a.key_dim)).astype(np.float32); k=rng.normal(0,.2,(*sh,a.key_dim)).astype(np.float32); v=rng.normal(0,.2,(*sh,a.value_dim)).astype(np.float32)
    g=rng.uniform(.90,.999,sh).astype(np.float32); beta=rng.uniform(.05,.95,sh).astype(np.float32); z=rng.normal(0,.2,(*sh,a.value_dim)).astype(np.float32)
    state=rng.normal(0,.05,(1,a.heads,a.value_dim,a.key_dim)).astype(np.float32)
    if a.model_dir:
        model,_=load(a.model_dir)
        mixer = model.language_model.model.layers[a.layer].linear_attn
        # MLX Qwen3.5 checkpoints have used both o_proj and out_proj for
        # this mathematically identical projection.  Resolve the model's
        # actual field instead of assuming one spelling.
        proj = None
        for name in ("o_proj", "out_proj", "output_projection"):
            candidate = getattr(mixer, name, None)
            if candidate is not None:
                proj = candidate
                break
        if proj is None:
            fields = sorted(k for k in dir(mixer) if "proj" in k.lower())
            raise AttributeError(f"cannot find GDN output projection; projection fields={fields}")
        w=np.asarray(mx.dequantize(proj.weight,proj.scales,proj.biases,proj.group_size,proj.bits,proj.mode,dtype=mx.float32)).T.astype(np.float32)
        if w.shape != (a.heads*a.value_dim,a.hidden): raise ValueError(f'output weight shape {w.shape} does not match expected {(a.heads*a.value_dim,a.hidden)}')
        # Build the same intermediate tensors as the model from a deterministic
        # hidden input.  This keeps the probe independent of tokenization while
        # exercising the real projection, convolution, normalization, and gate
        # parameters.
        hidden = (np.arange(a.tokens * a.hidden, dtype=np.int64) % 193 - 96).astype(np.float32).reshape(1, a.tokens, a.hidden) / 193.0
        x = mx.array(hidden)
        qkv0 = mixer.in_proj_qkv(x)
        conv_state = mx.zeros((1, mixer.conv_kernel_size - 1, mixer.conv_dim), dtype=x.dtype)
        conv = nn.silu(mixer.conv1d(mx.concatenate([conv_state, qkv0], axis=1)))
        q0, k0, v0 = [t.reshape(1, a.tokens, h, d) for t, h, d in zip(mx.split(conv, [mixer.key_dim, 2*mixer.key_dim], -1), [mixer.num_k_heads, mixer.num_k_heads, mixer.num_v_heads], [mixer.head_k_dim, mixer.head_k_dim, mixer.head_v_dim])]
        inv = mixer.head_k_dim ** -0.5
        q0 = (inv**2) * mx.fast.rms_norm(q0, None, 1e-6)
        k0 = inv * mx.fast.rms_norm(k0, None, 1e-6)
        repeat = mixer.num_v_heads // mixer.num_k_heads
        q0 = mx.repeat(q0, repeat, axis=2); k0 = mx.repeat(k0, repeat, axis=2)
        q = np.asarray(q0[0]); k = np.asarray(k0[0]); v = np.asarray(v0[0])
        z = np.asarray(mixer.in_proj_z(x).reshape(1, a.tokens, mixer.num_v_heads, mixer.head_v_dim)[0])
        b0 = mixer.in_proj_b(x)[0]
        a0 = mixer.in_proj_a(x)[0]
        beta = np.asarray(mx.sigmoid(b0))
        g = np.asarray(mx.exp(-mx.exp(mixer.A_log.astype(mx.float32)) * nn.softplus(a0 + mixer.dt_bias)))
        state = np.zeros((1, mixer.num_v_heads, mixer.head_v_dim, mixer.head_k_dim), dtype=np.float32)
        mx.eval(q0, k0, v0, mixer.in_proj_z(x), beta, g)
    else:
        w=rng.normal(0,.02,(a.heads*a.value_dim,a.hidden)).astype(np.float32)
    s=state[0].copy(); ys=[]
    for t in range(a.tokens):
        s*=g[t,:,None,None]; kv=np.einsum('hd,hvd->hv',k[t],s); d=(v[t]-kv)*beta[t,:,None]; s+=np.einsum('hd,hv->hvd',k[t],d); ys.append(np.einsum('hd,hvd->hv',q[t],s))
    recurrent=np.stack(ys); rms=np.mean(recurrent*recurrent,axis=-1,keepdims=True); gated=recurrent/np.sqrt(rms+1e-6)* (z/(1+np.exp(-z))); out=gated.reshape(a.tokens,-1)@w
    a.output.mkdir(parents=True,exist_ok=True)
    np.savez(a.output/'reference.npz',q=q,k=k,v=v,g=g,beta=beta,z=z,state=state,output_weight=w,output=out,final_state=s)
    for n,x in [('q',q),('k',k),('v',v),('g',g),('beta',beta),('z',z),('state',state),('output_weight',w)]: x.tofile(a.output/f'{n}.bin')
    print(f'saved={a.output}')
if __name__=='__main__': main()
