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

直接 Run（无参数）或传入 `--gui` 会打开 GPUI 桌面界面。模型页扫描配置目录，
可加载/卸载模型，也可从 Hugging Face 或 ModelScope 下载；后两者需要在项目
venv 中安装对应可选依赖。性能页运行 MiniMLX context matrix 并把完整输出保存在
`Reports/Performance/`。API 页启动/停止本机 OpenAI-compatible 服务
（`127.0.0.1:8000/v1`）。总览中的系统内存每 2 秒刷新；macOS 不提供可用的
通用 GPU 利用率接口，因此该项明确显示不可用，不显示模拟数据。

可选下载依赖：

```bash
./.venv/bin/pip install -e '.[download-huggingface]'
./.venv/bin/pip install -e '.[download-modelscope]'
```

OpenAI-compatible 服务支持 `/v1/models`、`/v1/chat/completions` 和
`/v1/completions`。文本流式响应遵循 SSE 格式，但当前以完整生成结果缓冲后发送；
Qwen3.5 视觉流式响应使用 runtime 的增量事件路径。

要运行已有命令行推理/基准流程，请给 binary 传入 CLI 参数，例如：

```text
--mode generate --prompt "Explain MLX" --max-tokens 32
```

CLI benchmark 仅测试 MiniMLX runtime，不会启动或调用 oMLX。
