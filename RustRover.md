# RustRover 运行 MiniMLX

RustRover 直接打开仓库根目录即可识别根目录的 `Cargo.toml` workspace。

先准备 MiniMLX 自己的 Python 环境：

```bash
zsh tools/bootstrap_venv.sh
cargo check --workspace
```

安装脚本默认使用清华 PyPI 镜像；如需切换源，可以覆盖
`PIP_INDEX_URL` 环境变量。

创建 Cargo Run Configuration：

- Package：`minimlx-gpui`
- Command：`run`
- Working directory：仓库根目录
- Program arguments：留空

程序默认读取：

```text
config/minimlx.local.json
```

该文件已经按当前机器的模型路径配置好。复制到其他机器时，复制
`config/minimlx.example.json` 为 `config/minimlx.local.json` 后修改路径即可。

临时参数仍然可以通过 Arguments 覆盖：

```text
--prompt "Explain MLX" --max-tokens 32
```

也可以通过环境变量 `MINIMLX_CONFIG` 指定另一份配置文件。

当前 binary 会启动 Python MLX worker，通过 Unix Domain Socket 发出一次
生成请求，并把生成文本和 tok/s 打印到 RustRover Run 窗口。

默认 `config/minimlx.local.json` 的 `mode` 是 `benchmark`，所以直接点击
Run 会执行与 oMLX 单请求界面对齐的测试：代码文本、上下文 4096 tokens、
生成 128 tokens、Quick warmup 32（一次 warm-up 请求，生成 32 tokens）。改为
`"mode": "generate"` 就会执行单次生成。GPUI 窗口会复用同一个
supervisor 和协议层接入。

当前 Run 先覆盖截图中的单请求基准。截图中的 batch=2/4 属于
continuous batching，必须由 Python scheduler 真正同时调度多个请求后再加入，
不会用串行调用伪装成 batch 结果。
