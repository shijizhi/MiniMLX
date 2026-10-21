use std::borrow::Cow;
use std::collections::HashMap;
use std::fs;
use std::ops::Range;
use std::path::Path;
use std::path::PathBuf;
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use gpui::{
    div, prelude::*, px, rgb, size, uniform_list, Action, AnyElement, App, Application,
    AssetSource, Bounds, ClipboardItem, Context, Entity, FocusHandle, Focusable, FontWeight,
    Render, SharedString, StatefulInteractiveElement, Subscription, Timer, UniformListScrollHandle,
    Window, WindowBounds, WindowOptions,
};
use gpui_component::button::{Button, ButtonCustomVariant, ButtonVariants};
use gpui_component::alert::Alert;
use gpui_component::checkbox::Checkbox;
use gpui_component::context_menu::ContextMenuExt;
use gpui_component::description_list::DescriptionList;
use gpui_component::dropdown::{Dropdown, DropdownEvent, DropdownItem, DropdownState};
use gpui_component::form::{form_field, v_form};
use gpui_component::indicator::Indicator;
use gpui_component::input::{InputEvent, InputState, TextInput};
use gpui_component::list::ListItem;
use gpui_component::progress::Progress;
use gpui_component::sidebar::{SidebarMenu, SidebarMenuItem};
use gpui_component::slider::{Slider, SliderEvent, SliderState};
use gpui_component::switch::Switch;
use gpui_component::tab::{Tab, TabBar};
use gpui_component::tag::Tag;
use gpui_component::text::TextView;
use gpui_component::theme::Theme;
use gpui_component::{ContextModal, Disableable, IndexPath, Root, Selectable, Sizable};
use serde::Deserialize;

use super::backend::{
    sort_model_search_results, validate_external_model_directory, Backend, ContextBenchmarkOptions,
    IncompleteDownload, ModelEntry, ModelInferenceOptions, ModelSearchResult, ModelSearchSort,
};
use serde_json::Value;

fn tr(language: &str, source: &str) -> String {
    if language != "zh-CN" {
        return source.to_string();
    }
    match source {
        "General" => "通用",
        "Server" => "服务器",
        "Network" => "网络",
        "Server & Network" => "服务器与网络",
        "Hugging Face endpoint" => "Hugging Face 镜像端点",
        "Proxy" => "代理",
        "Outbound HTTP routing. Blank proxy fields mean direct connection; system proxy variables are ignored." => "下载出站 HTTP 路由。代理项留空时直连；不会自动读取系统代理环境变量。",
        "HTTP proxy" => "HTTP 代理",
        "HTTPS proxy" => "HTTPS 代理",
        "No proxy" => "不使用代理",
        "Comma-separated hosts or CIDRs that bypass the proxy." => "逗号分隔的主机名或 CIDR 列表，这些地址将绕过代理。",
        "Used for HTTP requests, e.g. http://127.0.0.1:7890" => "用于 HTTP 请求，例如 http://127.0.0.1:7890",
        "Used for HTTPS requests, e.g. http://127.0.0.1:7890" => "用于 HTTPS 请求，例如 http://127.0.0.1:7890",
        "TLS" => "TLS",
        "Optional custom certificate authority for HTTPS connections." => "HTTPS 连接使用的自定义证书颁发机构（可选）。",
        "CA bundle" => "CA 证书包",
        "PEM certificate bundle path; blank uses system trust." => "PEM 格式证书包路径；留空时使用系统信任存储。",
        "Not set" => "未设置",
        "Used for model metadata and file resolution. Xet/CAS is disabled; the Hub may still redirect file bytes to its storage host." => "用于模型信息和文件解析。已关闭 Xet/CAS；Hub 仍可能将权重数据重定向到其文件存储主机。",
        "Default: https://huggingface.co" => "默认：https://huggingface.co",
        "Security" => "安全",
        "Inference defaults" => "推理默认值",
        "Startup" => "启动",
        "Storage" => "存储",
        "Logs" => "日志",
        "Overview" => "概览",
        "Models" => "模型",
        "Back to models" => "返回模型列表",
        "Model configuration" => "模型配置",
        "Basic" => "基本",
        "Advanced" => "高级",
        "General specification" => "基础身份与规格 (General Specification)",
        "Set the model identifier and per-request token limits." => "定义运行时展示标识与窗口长度",
        "Maximum generated tokens per response." => "单次回复的最大生成长度。",
        "Presets" => "预设",
        "KV cache maximum capacity." => "KV Cache 可容纳的最大上下文长度。",
        "Use a compatible draft model to accelerate text generation." => "配置草稿加速模型以提升 Apple Silicon 端侧 Token 输出速度",
        "Speculative Decoding" => "推测解码 (Speculative Decoding)",
        "DFlash configured" => "已配置 DFlash",
        "MTP configured" => "已配置 MTP",
        "Text generation only. Availability depends on target and draft model compatibility." => "仅支持文本生成；是否可用取决于目标模型与草稿模型的兼容性。",
        "Sampling parameters" => "采样精调参数 (Sampling Parameters)",
        "Control randomness and token selection during generation." => "调整模型发散度与概率截断策略",
        "Higher values increase output variety." => "数值越高，输出越具多样性。",
        "Cumulative probability cutoff." => "按累计概率筛选候选 Token。",
        "Set to 0 to disable." => "设为 0 时不启用。",
        "Penalize tokens already present in output." => "降低已生成 Token 再次出现的概率。",
        "Encourage new topics." => "提高生成新主题的倾向。",
        "Advanced sampling" => "高级采样",
        "Additional controls for API and template behavior." => "配置 API 采样覆盖与聊天模板参数。",
        "Blank fields use application defaults" => "留空项沿用应用默认值",
        "Max output tokens" => "最大输出 Token",
        "Min P" => "Min P",
        "Presence penalty" => "存在惩罚",
        "Frequency penalty" => "频率惩罚",
        "Force sampling" => "强制采样",
        "Ignore sampling values sent by API clients." => "忽略 API 请求携带的采样参数，始终使用此模型配置。",
        "Chat template parameters" => "聊天模板参数",
        "JSON object forwarded to the model chat template." => "以 JSON 对象传入模型聊天模板。",
        "Reasoning" => "思考模式",
        "Thinking budget" => "思考预算",
        "Tool-result token limit" => "工具结果 Token 上限",
        "Reasoning parser" => "思考解析器",
        "Keep in memory" => "常驻内存",
        "Favorite" => "收藏",
        "Trust remote code" => "信任远程代码",
        "Lightning MTP" => "Lightning MTP",
        "DFlash" => "DFlash",
        "VLM MTP" => "VLM MTP",
        "Speculative decoding" => "推测解码",
        "Draft model" => "草稿模型",
        "Off · use standard decoding" => "关闭 · 使用标准解码",
        "Select a compatible MTP or DFlash draft model to accelerate text generation." => "选择兼容的 MTP 或 DFlash 草稿模型以加速文本生成。",
        "No compatible speculative draft installed" => "尚未安装兼容的推测解码草稿模型",
        "MTP draft model is unavailable" => "MTP 草稿模型不可用",
        "MTP draft · attach to a target model" => "MTP 草稿 · 请绑定到目标模型",
        "DFlash draft · attach to a target model" => "DFlash 草稿 · 请绑定到目标模型",
        "Draft" => "草稿模型",
        "Text-only generation. MTP and DFlash support depends on the target/draft pairing." => "仅支持文本生成；MTP 和 DFlash 是否可用取决于目标模型与草稿模型的配对。",
        "Qwen INT8 activation prefill" => "Qwen INT8 激活预填充",
        "Qwen ANE prefill" => "Qwen ANE 预填充",
        "Tune ANE split" => "调整 ANE 分配",
        "TurboQuant KV" => "TurboQuant KV 缓存",
        "SpecPrefill" => "SpecPrefill",
        "Planned" => "待实现",
        "Model type" => "模型类型",
        "Architecture" => "架构",
        "Quantization" => "量化",
        "Disk usage" => "磁盘占用",
        "Inherit" => "继承默认值",
        "Performance Tests" => "性能测试",
        "Contexts" => "测试上下文",
        "Warm-up tokens" => "预热 Token 数",
        "Runs per context" => "每种上下文重复次数",
        "Output tokens" => "生成 Token 数",
        "Prefill step" => "预填充步长",
        "Ignore EOS, fixed output length" => "忽略 EOS，固定生成长度",
        "Results" => "结果",
        "rows" => "行",
        "Log" => "日志",
        "Detailed report" => "详细报告",
        "Copy log" => "复制日志",
        "Copy report" => "复制报告",
        "Export CSV" => "导出 CSV",
        "Report exported" => "报告已导出",
        "Benchmark v2.4" => "基准测试 v2.4",
        "Measure local model prefill throughput, decode latency, and memory usage." => "测量本地模型的预填充吞吐、解码延迟和内存占用。",
        "All parameters are applied to the local inference engine." => "所有参数都会实时同步至本地推理引擎。",
        "Pre-run tokens" => "预热 Token 数",
        "Pre-run reduces cache cold-start effects" => "预热以消除缓存冷启动影响",
        "Times per context" => "每种上下文重复次数",
        "Repeated runs provide a more stable median" => "多次计算以获取更稳定结果",
        "Decode phase output token limit" => "解码阶段总 token 长度",
        "Chunked-prefill window size" => "Chunked-Prefill 窗口大小",
        "Selected contexts" => "已选上下文长度",
        "context lengths selected for the performance sweep" => "个上下文长度用于性能阶梯测试",
        "tokens" => "tokens",
        "times" => "次",
        "step" => "step",
        "Performance results" => "性能结果",
        "No context results found in this report." => "此报告中没有上下文测试结果。",
        "History archive" => "历史归档",
        "Ready" => "设备已就绪",
        "Metal API active" => "Metal API 已启用",
        "Idle" => "就绪（空闲）",
        "Benchmark options" => "测试参数",
        "Select at least one context length." => "至少选择一个上下文长度。",
        "API Server" => "API 服务",
        "Settings" => "设置",
        "Local model operations" => "本地模型管理",
        "WORKSPACE" => "工作区",
        "Local MLX runtime" => "本地 MLX 运行时",
        "Service running" => "服务运行中",
        "Runtime ready" => "运行时就绪",
        "Local runtime" => "本地运行时",
        "Apple Silicon" => "Apple 芯片",
        "Monitor local models, unified memory, and inference throughput." => "监控本地模型、统一内存和推理吞吐。",
        "Inference Service" => "推理服务",
        "Stopped" => "已停止",
        "Running" => "运行中",
        "Only models in the MiniMLX model list can be started" => "只能启动 MiniMLX 模型列表中的模型",
        "model is not available in the MiniMLX model list" => "该模型不在 MiniMLX 模型列表中",
        "Listening" => "正在监听",
        "ACTIVE MODEL" => "当前模型",
        "No model loaded" => "未加载模型",
        "No models loaded" => "没有已加载的模型",
        "No model configured" => "未配置模型",
        "MTP draft configured" => "已配置 MTP 草稿模型",
        "MTP is not configured for this model" => "此模型未配置 MTP",
        "Testing…" => "测试中…",
        "Preparing the model and measuring selected context lengths." => "正在准备模型并测试所选上下文长度。",
        "Benchmark failed" => "性能测试失败",
        "History reports" => "历史报告",
        "Select model" => "选择模型",
        "Choose a model to start the service" => "选择要启动服务的模型",
        "Select one of your MiniMLX models. The service will start after you choose." => "请选择一个 MiniMLX 模型；选择后将启动服务。",
        "No runnable MiniMLX models are available" => "没有可启动的 MiniMLX 模型",
        "Download or add a model in Models before starting the service." => "请先在“模型”页面下载或添加模型，再启动服务。",
        "Go to Models" => "前往模型",
        "Open API Docs" => "打开 API 文档",
        "Copy" => "复制",
        "API endpoint copied" => "API 地址已复制",
        "Error copied to clipboard" => "错误信息已复制",
        "System Resources" => "系统资源",
        "Live snapshot · macOS" => "实时状态 · macOS",
        "Memory in use" => "内存使用量",
        "GPU utilization" => "GPU 使用率",
        "MiniMLX · serving requests" => "MiniMLX · 正在处理请求",
        "MiniMLX · service stopped" => "MiniMLX · 服务已停止",
        "Not exposed by macOS telemetry" => "macOS 未提供此遥测数据",
        "Metal device · Apple Silicon" => "Metal 设备 · Apple 芯片",
        "Current server process" => "当前服务进程",
        "Start service to collect statistics" => "启动服务后开始收集统计数据",
        "Service Statistics" => "服务统计",
        "Requests" => "请求数",
        "Active" => "处理中",
        "Prompt tokens" => "提示词 Token",
        "Generated tokens" => "生成 Token",
        "Since server start" => "自服务启动以来",
        "In-flight requests" => "正在处理的请求",
        "Processed input" => "已处理输入",
        "Completion output" => "生成输出",
        "Avg. prefill" => "平均提示词处理速度",
        "Avg. generation" => "平均生成速度",
        "Prompt tokens / second" => "提示词 Token / 秒",
        "Generated tokens / second" => "生成 Token / 秒",
        "Uptime" => "运行时长",
        "Failed requests" => "失败请求",
        "Loaded Models" => "已加载模型",
        "Manage models" => "管理模型",
        "Configure model" => "配置模型",
        "Model inference settings" => "模型推理设置",
        "Model alias" => "模型别名",
        "API model ID · blank uses repository name" => "API 模型 ID · 留空使用仓库名称",
        "Generation" => "生成",
        "Maximum output tokens · blank uses app default" => "最大输出 Token · 留空使用应用默认值",
        "Context window · blank uses app default" => "上下文窗口 · 留空使用应用默认值",
        "Temperature · blank uses app default" => "Temperature · 留空使用应用默认值",
        "Top P · blank uses app default" => "Top P · 留空使用应用默认值",
        "Top K · blank uses app default" => "Top K · 留空使用应用默认值",
        "Sampling" => "采样",
        "Repetition penalty · blank uses app default" => "重复惩罚 · 留空使用应用默认值",
        "Restore defaults" => "恢复默认",
        "Save" => "保存",
        "Cancel" => "取消",
        "Model" => "模型",
        "Model name and architecture" => "模型名称与架构",
        "Filter models…" => "筛选模型…",
        "Filter downloaded models…" => "筛选已下载模型…",
        "Model information" => "模型信息",
        "Copy model name" => "复制模型名称",
        "MLX native" => "MLX 框架原生",
        "Instruct" => "指令微调",
        "4-bit quantized" => "4-bit 量化",
        "quantized" => "量化",
        "Vision" => "视觉",
        "MoE" => "混合专家",
        "Show in Finder" => "在访达中显示",
        "Unified memory" => "统一内存",
        "Ready on disk (verified)" => "已验证，可从磁盘加载",
        "Status" => "状态",
        "DFlash draft" => "DFlash 草稿",
        "MTP draft" => "MTP 草稿",
        "Draft checkpoints are attached from a target model's configuration." => "草稿检查点需在目标模型的配置中关联。",
        "No models installed" => "尚未安装模型",
        "No matching installed models" => "没有匹配的已下载模型",
        "Download or add a model to manage it here." => "下载或添加模型后，即可在此处管理。",
        "File location" => "文件位置",
        "Parameters" => "参数",
        "Model family" => "模型类型",
        "No model selected" => "未选择模型",
        "Select a model to inspect its details and actions." => "选择模型以查看详细信息和可用操作。",
        "Delete model" => "删除模型",
        "Delete this model?" => "删除这个模型？",
        "This permanently removes the model files from disk. This action cannot be undone." => "这会永久删除磁盘上的模型文件，且无法撤销。",
        "Delete permanently" => "永久删除",
        "Format / quant" => "格式 / 量化",
        "Disk" => "磁盘占用",
        "Service" => "服务状态",
        "Actions" => "操作",
        "Not declared" => "未声明",
        "Loaded · ready" => "已加载 · 就绪",
        "Not loaded" => "未加载",
        "Local checkpoint" => "本地模型文件",
        "Load a model from Models" => "请从“模型”页面加载模型",
        "Unload" => "卸载",
        "Unloading model…" => "正在卸载模型…",
        "Model unloaded" => "模型已卸载",
        "Recent Activity" => "最近活动",
        "Recent requests" => "最近请求",
        "No requests in this server session yet." => "当前服务会话尚无请求。",
        "Requests will appear here when the service is running." => "服务启动后，请求会显示在这里。",
        "Inference request" => "推理请求",
        "MiniMLX · updates every 2 seconds" => "MiniMLX · 每 2 秒更新",
        "Discover, import, and manage local model artifacts." => "发现、导入和管理本地模型文件。",
        "Search models by repository or keyword" => "搜索模型仓库或关键词",
        "Searching…" => "正在搜索…",
        "No matching models" => "没有找到匹配的模型",
        "Model card" => "模型卡片",
        "Enter a valid model repository ID before downloading." => "下载前请输入有效的模型仓库 ID。",
        "Loading…" => "正在加载…",
        "Select a model from the search results before downloading." => "请先从搜索结果中选择模型，再开始下载。",
        "Downloaded" => "已下载",
        "Discover" => "发现模型",
        "Downloads" => "下载",
        "Name" => "名称",
        "Size" => "大小",
        "Likes" => "点赞数",
        "Sort by" => "排序",
        "Download count" => "下载量",
        "Location" => "位置",
        "Runtime" => "运行时",
        "Memory guard: monitoring" => "内存保护监控中",
        "Memory pressure: new requests paused" => "内存压力较高：已暂停新请求",
        "Memory status unavailable" => "内存状态暂不可用",
        "Memory guard: checking" => "正在检查内存状态",
        "Memory guard: waiting" => "服务启动后启用内存保护",
        "Load" => "加载",
        "Loaded" => "已加载",
        "On disk" => "在磁盘上",
        "Add from model hub" => "从模型仓库添加",
        "Search for DFlash drafts by model or publisher; downloads use the selected repository." => "可按模型名或发布者搜索 DFlash 草稿模型；下载所选仓库。",
        "Install location" => "安装位置",
        "Repository ID" => "仓库 ID",
        "Download Model" => "下载模型",
        "Add Model" => "添加模型",
        "Refresh" => "刷新",
        "Benchmark" => "性能测试",
        "Latest report" => "最新报告",
        "report(s)" => "份报告",
        "Download queue" => "下载队列",
        "Resume" => "继续下载",
        "Incomplete download" => "未完成下载",
        "No incomplete downloads" => "没有未完成的下载",
        "Delete incomplete download" => "删除未完成下载",
        "Pause download" => "暂停下载",
        "Delete" => "删除",
        "Run reproducible tests on the MiniMLX runtime." => "在 MiniMLX 运行时执行可复现的性能测试。",
        "Test configuration" => "测试配置",
        "Suite" => "测试套件",
        "Context matrix · 1K / 4K / 8K / 16K" => "上下文矩阵 · 1K / 4K / 8K / 16K",
        "Warm-up / repetitions" => "预热 / 重复次数",
        "32-token warm-up · 3 runs per context · median" => "预热 32 Token · 每种上下文测试 3 次 · 取中位数",
        "MiniMLX only" => "仅测试 MiniMLX",
        "No report yet. Run the MiniMLX context matrix to create one." => "暂无报告。运行 MiniMLX 上下文矩阵测试以生成报告。",
        "Run MiniMLX Test" => "运行 MiniMLX 测试",
        "Configure and monitor the local inference endpoint." => "配置并监控本地推理接口。",
        "Server status" => "服务状态",
        "Start Service" => "启动服务",
        "Stop Service" => "停止服务",
        "Base URL" => "基础 URL",
        "Routes" => "接口路由",
        "Compatibility" => "兼容性",
        "Network access" => "网络访问",
        "Text streaming" => "文本流式输出",
        "Loopback only · unauthenticated local API" => "仅本机回环访问 · 未启用 API 密钥验证",
        "OpenAI Chat Completions · text and local/data-URI images" => "兼容 OpenAI Chat Completions · 文本及本地 / data URI 图片",
        "SSE-compatible · buffered until generation completes" => "兼容 SSE · 生成完成后返回缓冲结果",
        "GET /v1/models · GET /metrics · POST /v1/chat/completions" => "GET /v1/models · GET /metrics · POST /v1/chat/completions",
        "Local API listener and service endpoints." => "本地 API 监听地址与服务接口。",
        "Require Bearer token or x-api-key for API requests" => "API 请求必须提供 Bearer Token 或 x-api-key。",
        "The API key is stored in macOS Keychain, not in the settings file. LAN binding is rejected until a key exists." => "API 密钥保存在 macOS 钥匙串中，不会写入设置文件。生成密钥后才能启用局域网访问。",
        "Expose the service on all network interfaces; API-key authentication is required" => "在所有网络接口上开放服务；必须启用 API 密钥验证。",
        "Default 18765 · service restarts when applied" => "默认端口 18765 · 应用后重启服务",
        "Chat Completions and model listing" => "Chat Completions 与模型列表",
        "POST /v1/messages · text and image requests; tool use is not advertised" => "POST /v1/messages · 支持文本和图片请求；不声明工具调用支持",
        "GET /health" => "GET /health",
        "GET /metrics · Prometheus text exposition" => "GET /metrics · Prometheus 文本格式",
        "Changes apply to the next server start. If the service is running, Apply restarts it with the new settings." => "更改将在下次启动服务时生效。若服务正在运行，点击“应用”会使用新设置重启服务。",
        "Default text context cap · oldest prompt tokens are trimmed when exceeded" => "默认文本上下文上限 · 超出时从最早的提示词开始裁剪",
        "Default 32768 · request parameters can override" => "默认 32768 · 可由单次请求参数覆盖",
        "Sampling randomness · 0 to 2" => "采样随机性 · 范围 0 到 2",
        "Nucleus sampling · 0 to 1" => "核采样 · 范围 0 到 1",
        "Candidate limit · 0 disables Top K" => "候选数量上限 · 0 表示关闭 Top K",
        "Penalty for repeated tokens · 1 disables penalty" => "重复 Token 惩罚 · 1 表示不惩罚",
        "Starts the configured model service when the desktop app launches; does not add a macOS login item" => "启动 MiniMLX 时启动已配置的模型服务；不会添加 macOS 登录项。",
        "Stores MiniMLX runtime data such as token-usage history; model download/search paths are listed below" => "保存 Token 使用历史等 MiniMLX 运行数据；模型下载与搜索目录在下方配置。",
        "Store hourly per-model prompt and generated token totals for 400 days; disabling keeps existing history" => "按模型每小时记录提示词和生成 Token 数，保留 400 天；关闭后仍保留已有记录。",
        "SQLite · aggregate token totals only, not prompt or response text" => "SQLite · 仅保存 Token 汇总，不保存提示词或回复内容。",
        "API process output is inherited by the app's launch environment" => "API 进程输出写入启动应用的终端或日志环境。",
        "Start API service when MiniMLX opens" => "打开 MiniMLX 时启动 API 服务",
        "Base path" => "基础路径",
        "Stores runtime data; models use the MiniMLX library and explicitly added external folders" => "保存 MiniMLX 运行数据；模型使用 MiniMLX 模型库和明确添加的外部目录。",
        "MiniMLX model library" => "MiniMLX 模型库",
        "External model folders" => "外部模型目录",
        "Common model folders" => "常见模型目录",
        "oMLX" => "oMLX",
        "LM Studio" => "LM Studio",
        "MiniMLX library · download destination" => "MiniMLX 自有目录 · 下载位置",
        "External folder · referenced in place" => "外部目录 · 从原位置引用",
        "Enter an external model folder path" => "输入外部模型目录路径",
        "Add external folder" => "添加外部目录",
        "The managed MiniMLX folder is the download destination. Only explicitly added external folders are searched; .cache directories are never scanned." => "MiniMLX 自有目录是模型下载位置。仅扫描明确添加的外部目录，不会扫描 .cache。",
        "External folders are referenced in place; files are not copied or moved." => "外部目录以引用方式使用，不会复制或移动其中的文件。",
        "No supported model found in that folder. Add a folder containing a MiniMLX-compatible model." => "该目录中没有受支持的模型。请选择包含 MiniMLX 兼容模型的目录。",
        "Model folders inside .cache cannot be added." => "不能添加位于 .cache 中的模型目录。",
        "This external model folder is already added." => "此外部模型目录已添加。",
        "Health endpoint" => "健康检查接口",
        "Unauthenticated" => "无需验证",
        "GET /health only; all inference, model, usage and metrics routes require the key when enabled" => "仅 GET /health 无需验证；启用验证后，推理、模型、使用统计和指标接口均需密钥。",
        "Transport" => "传输协议",
        "For access beyond a trusted local network, use a TLS reverse proxy" => "如需在可信局域网之外访问，请使用 TLS 反向代理。",
        "HTTP" => "HTTP",
        "The API key is stored in macOS Keychain, not in the settings file." => "API 密钥存储在 macOS 钥匙串中，不会写入设置文件。",
        "Language saved" => "界面语言已保存",
        "Configure the local inference service and runtime preferences." => "配置本地推理服务和运行时选项。",
        "Search settings" => "搜索设置",
        "Unsaved changes" => "有未保存的更改",
        "Settings are saved in your user profile" => "设置保存在当前用户配置中",
        "Reset" => "重置",
        "Apply" => "应用",
        "Listen on LAN" => "允许局域网访问",
        "Port" => "端口",
        "API key authentication" => "API 密钥验证",
        "OpenAI-compatible API" => "兼容 OpenAI 的 API",
        "Anthropic Messages" => "Anthropic Messages",
        "Health check" => "健康检查",
        "Prometheus metrics" => "Prometheus 指标",
        "History database" => "历史记录数据库",
        "Service logs" => "服务日志",
        "MiniMLX Run log" => "MiniMLX 运行日志",
        "Context window" => "上下文窗口",
        "Maximum output tokens" => "最大输出 Token 数",
        "Temperature" => "温度",
        "Top P" => "Top P",
        "Top K" => "Top K",
        "Repetition penalty" => "重复惩罚",
        "Log level" => "日志级别",
        "Record usage history" => "记录使用历史",
        "API key" => "API 密钥",
        "Stored in macOS Keychain" => "已存储在 macOS 钥匙串",
        "No key generated" => "尚未生成密钥",
        "Regenerate" => "重新生成",
        "Generate key" => "生成密钥",
        "Enter a model directory path" => "输入模型目录路径",
        "Add" => "添加",
        "Remove" => "移除",
        "On" => "开启",
        "Off" => "关闭",
        "Language" => "界面语言",
        "English" => "英语",
        "简体中文" => "简体中文",
        "System" => "系统",
        "Choose the language used by the MiniMLX interface." => "选择 MiniMLX 界面显示的语言。",
        "Could not save language preference" => "无法保存界面语言设置",
        "Could not create API key" => "无法创建 API 密钥",
        "runtime lock poisoned" => "运行时状态锁不可用",
        "Inventory refresh failed" => "刷新模型列表失败",
        "Found" => "已发现",
        "local model(s)" => "个本地模型",
        "Saved" => "已保存",
        "Runtime stopped" => "运行时已停止",
        "API service started on app launch" => "应用启动时已启动 API 服务",
        "API server started" => "API 服务已启动",
        "API server stopped" => "API 服务已停止",
        "Benchmark running · this may take several minutes…" => "性能测试正在运行 · 可能需要几分钟…",
        "Benchmark complete · report saved" => "性能测试完成 · 报告已保存",
        "Loading model…" => "正在加载模型…",
        "Refreshing model inventory…" => "正在刷新模型列表…",
        "Settings applied" => "设置已应用",
        "Applying settings and restarting the service…" => "正在应用设置并重启服务…",
        "Settings saved · used the next time the service starts" => "设置已保存 · 下次启动服务时生效",
        "Local model artifacts" => "本地模型文件",
        "Downloading" => "正在下载",
        "Download complete" => "下载完成",
        "Download failed" => "下载失败",
        "Listening ·" => "正在监听 ·",
        "Port must be a number between 1024 and 65535" => "端口必须为 1024 到 65535 之间的数字",
        "Maximum tokens must be a positive whole number" => "最大 Token 数必须为正整数",
        "Context window must be a positive whole number" => "上下文窗口必须为正整数",
        "Temperature must be a finite, non-negative number" => "温度必须是有限的非负数",
        "Top P must be a number from 0 to 1" => "Top P 必须是 0 到 1 之间的数字",
        "Top K must be a non-negative whole number" => "Top K 必须是非负整数",
        "Repetition penalty must be greater than 0" => "重复惩罚必须大于 0",
        "One or more values are outside the supported range" => "一个或多个值超出了允许范围",
        "API-key authentication requires a key. Generate one in Security first." => "启用 API 密钥验证前，请先在“安全”中生成密钥。",
        "LAN access requires an API key. Generate one in Security first." => "启用局域网访问前，请先在“安全”中生成 API 密钥。",
        "Keep at least one model directory configured" => "请至少保留一个模型目录。",
        "Enter a model directory path first" => "请先输入模型目录路径。",
        "New API key saved to Keychain and copied to clipboard" => "新的 API 密钥已保存到钥匙串并复制到剪贴板。",
        other => other,
    }
    .to_string()
}

fn is_speculative_draft_entry(entry: &ModelEntry) -> bool {
    entry.model_type.as_deref() == Some("qwen3_5_mtp")
        || entry
            .architecture
            .as_deref()
            .is_some_and(|architecture| architecture.to_ascii_lowercase().contains("dflash"))
}

fn qwen_mtp_pair_compatible(target_path: &str, draft_path: &str) -> bool {
    let read_config = |path: &str| -> Option<Value> {
        fs::read_to_string(Path::new(path).join("config.json"))
            .ok()
            .and_then(|contents| serde_json::from_str(&contents).ok())
    };
    let Some(target) = read_config(target_path) else {
        return false;
    };
    let Some(draft) = read_config(draft_path) else {
        return false;
    };
    let target_type = target
        .get("model_type")
        .and_then(Value::as_str)
        .unwrap_or_default()
        .replace('-', "_")
        .to_ascii_lowercase();
    if !["qwen3_5", "qwen3_6", "qwen3_8"]
        .iter()
        .any(|family| target_type.starts_with(family))
        || draft.get("model_type").and_then(Value::as_str) != Some("qwen3_5_mtp")
    {
        return false;
    }
    let target_text = target.get("text_config").unwrap_or(&target);
    let draft_text = draft.get("text_config").unwrap_or(&draft);
    ["hidden_size", "vocab_size"].iter().all(|key| {
        match (target_text.get(key), draft_text.get(key)) {
            (Some(target_value), Some(draft_value)) => target_value == draft_value,
            _ => true,
        }
    })
}

fn qwen_dflash_pair_compatible(target_path: &str, draft_path: &str) -> bool {
    let read_config = |path: &str| -> Option<Value> {
        fs::read_to_string(Path::new(path).join("config.json"))
            .ok()
            .and_then(|contents| serde_json::from_str(&contents).ok())
    };
    let Some(target) = read_config(target_path) else { return false };
    let Some(draft) = read_config(draft_path) else { return false };
    let target_type = target.get("model_type").and_then(Value::as_str).unwrap_or_default()
        .replace('-', "_").to_ascii_lowercase();
    let draft_facts = format!("{} {}",
        draft.get("model_type").and_then(Value::as_str).unwrap_or_default(),
        draft.get("architectures").map(Value::to_string).unwrap_or_default()
    ).to_ascii_lowercase();
    if !["qwen3_5", "qwen3_6", "qwen3_8"].iter().any(|family| target_type.starts_with(family))
        || !draft_facts.contains("dflash")
    {
        return false;
    }
    let target_text = target.get("text_config").unwrap_or(&target);
    ["hidden_size", "vocab_size"].iter().all(|key| {
        match (target_text.get(key), draft.get(key)) {
            (Some(target_value), Some(draft_value)) => target_value == draft_value,
            _ => true,
        }
    })
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum SettingsCategory {
    General,
    Server,
    Network,
    Security,
    Inference,
    Startup,
    Storage,
    Logs,
}

#[derive(Clone, Copy)]
enum NumericSetting {
    Port,
    MaxTokens,
    ContextWindow,
    Temperature,
    TopP,
    TopK,
    RepetitionPenalty,
}

#[derive(Clone, Copy)]
enum BooleanSetting {
    BindLan,
    Authentication,
    StartService,
    UsageHistory,
}

#[derive(Clone, Copy)]
enum DownloadTextSetting {
    HttpProxy,
    HttpsProxy,
    NoProxy,
    CaBundle,
}

impl SettingsCategory {
    const ALL: [Self; 8] = [
        Self::General,
        Self::Server,
        Self::Network,
        Self::Security,
        Self::Inference,
        Self::Startup,
        Self::Storage,
        Self::Logs,
    ];

    fn title(self, language: &str) -> String {
        let english = match self {
            Self::General => "General",
            Self::Server => "Server",
            Self::Network => "Network",
            Self::Security => "Security",
            Self::Inference => "Inference defaults",
            Self::Startup => "Startup",
            Self::Storage => "Storage",
            Self::Logs => "Logs",
        };
        tr(language, english)
    }
}

#[derive(Clone, serde::Deserialize, serde::Serialize)]
#[serde(default)]
struct DesktopSettings {
    language: String,
    port: u16,
    max_tokens: u32,
    max_context_tokens: u32,
    bind_lan: bool,
    auth_enabled: bool,
    temperature: f32,
    top_p: f32,
    top_k: u32,
    repetition_penalty: f32,
    start_service_on_launch: bool,
    log_level: String,
    record_usage_history: bool,
    data_path: String,
    model_directories: Vec<String>,
    model_inference: HashMap<String, ModelInferenceOptions>,
    huggingface_endpoint: String,
    http_proxy: String,
    https_proxy: String,
    no_proxy: String,
    ca_bundle: String,
}

impl Default for DesktopSettings {
    fn default() -> Self {
        Self {
            language: "en".into(),
            port: 18765,
            max_tokens: 32768,
            max_context_tokens: 32768,
            bind_lan: false,
            auth_enabled: false,
            temperature: 1.0,
            top_p: 0.95,
            top_k: 0,
            repetition_penalty: 1.0,
            start_service_on_launch: false,
            log_level: "INFO".into(),
            record_usage_history: true,
            data_path: std::env::var_os("HOME")
                .map(PathBuf::from)
                .unwrap_or_else(|| PathBuf::from("."))
                .join(".minimlx")
                .display()
                .to_string(),
            model_directories: Vec::new(),
            model_inference: HashMap::new(),
            huggingface_endpoint: "https://huggingface.co".into(),
            http_proxy: String::new(),
            https_proxy: String::new(),
            no_proxy: String::new(),
            ca_bundle: String::new(),
        }
    }
}

fn desktop_settings_path() -> Option<PathBuf> {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .map(|home| home.join(".minimlx").join("settings.json"))
}

fn managed_models_directory() -> PathBuf {
    std::env::var_os("HOME")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".minimlx/models")
}

fn expand_user_directory(path: &str) -> PathBuf {
    if let Some(path) = path.strip_prefix("~/") {
        std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_else(|| PathBuf::from("."))
            .join(path)
    } else {
        PathBuf::from(path)
    }
}

fn normalized_model_settings_key(path: &str) -> String {
    expand_user_directory(path)
        .canonicalize()
        .unwrap_or_else(|_| expand_user_directory(path))
        .display()
        .to_string()
}

fn same_model_path(left: &str, right: &str) -> bool {
    left == right || normalized_model_settings_key(left) == normalized_model_settings_key(right)
}

fn model_inference_options_for_path<'a>(
    settings: &'a DesktopSettings,
    model_path: &str,
) -> Option<&'a ModelInferenceOptions> {
    if let Some(options) = settings.model_inference.get(model_path) {
        return Some(options);
    }
    settings
        .model_inference
        .iter()
        .find_map(|(configured_path, options)| {
            same_model_path(configured_path, model_path).then_some(options)
        })
}

fn model_settings_saved_message(options: &ModelInferenceOptions) -> String {
    if let Some(path) = options.dflash_draft_model.as_deref() {
        let draft = Path::new(path).file_name().and_then(|name| name.to_str()).unwrap_or(path);
        return format!("Model settings saved · DFlash enabled ({draft})");
    }
    match options.mtp_draft_model.as_deref() {
        Some(path) => {
            let draft = Path::new(path)
                .file_name()
                .and_then(|name| name.to_str())
                .unwrap_or(path);
            format!("Model settings saved · MTP enabled ({draft})")
        }
        None => "Model settings saved · MTP disabled".to_string(),
    }
}

fn valid_temperature(value: f32) -> bool {
    value.is_finite() && value >= 0.0
}

fn load_desktop_settings() -> DesktopSettings {
    let mut settings: DesktopSettings = desktop_settings_path()
        .and_then(|path| fs::read_to_string(path).ok())
        .and_then(|contents| serde_json::from_str(&contents).ok())
        .unwrap_or_default();
    if let Some(path) = settings.data_path.strip_prefix("~/") {
        settings.data_path = std::env::var_os("HOME")
            .map(PathBuf::from)
            .unwrap_or_default()
            .join(path)
            .display()
            .to_string();
    }
    if !(1024..=65535).contains(&settings.port)
        || !(1..=1_000_000).contains(&settings.max_tokens)
        || !(1..=1_000_000).contains(&settings.max_context_tokens)
        || !settings.temperature.is_finite()
        || !valid_temperature(settings.temperature)
        || !settings.top_p.is_finite()
        || !(0.0..=1.0).contains(&settings.top_p)
        || !settings.repetition_penalty.is_finite()
        || settings.repetition_penalty <= 0.0
        || !["DEBUG", "INFO", "WARNING", "ERROR"].contains(&settings.log_level.as_str())
        || !["en", "zh-CN"].contains(&settings.language.as_str())
        || !(settings.huggingface_endpoint.starts_with("https://")
            || settings.huggingface_endpoint.starts_with("http://"))
        || settings
            .huggingface_endpoint
            .chars()
            .any(char::is_whitespace)
    {
        DesktopSettings::default()
    } else {
        settings.model_inference.retain(|_, options| {
            options.alias.as_ref().is_none_or(|value| {
                !value.is_empty() && value.len() <= 128 && !value.chars().any(char::is_whitespace)
            }) && options
                .max_tokens
                .is_none_or(|value| (1..=1_000_000).contains(&value))
                && options
                    .max_context_tokens
                    .is_none_or(|value| (1..=1_000_000).contains(&value))
                && options
                    .temperature
                    .is_none_or(valid_temperature)
                && options
                    .top_p
                    .is_none_or(|value| value.is_finite() && (0.0..=1.0).contains(&value))
                && options
                    .repetition_penalty
                    .is_none_or(|value| value.is_finite() && value > 0.0)
        });
        settings
    }
}

fn keychain_api_key() -> Option<String> {
    Command::new("/usr/bin/security")
        .args([
            "find-generic-password",
            "-a",
            "minimlx",
            "-s",
            "MiniMLX.APIKey",
            "-w",
        ])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|key| !key.is_empty())
}

fn store_keychain_api_key(key: &str) -> Result<(), String> {
    let output = Command::new("/usr/bin/security")
        .args([
            "add-generic-password",
            "-U",
            "-a",
            "minimlx",
            "-s",
            "MiniMLX.APIKey",
            "-w",
            key,
        ])
        .output()
        .map_err(|error| error.to_string())?;
    if output.status.success() {
        Ok(())
    } else {
        Err(String::from_utf8_lossy(&output.stderr).trim().to_string())
    }
}

fn persist_desktop_settings(settings: &DesktopSettings) -> Result<(), String> {
    let path = desktop_settings_path().ok_or("home directory is unavailable")?;
    let parent = path
        .parent()
        .ok_or("settings path has no parent directory")?;
    fs::create_dir_all(parent).map_err(|error| error.to_string())?;
    let contents = serde_json::to_string_pretty(settings).map_err(|error| error.to_string())?;
    fs::write(path, contents).map_err(|error| error.to_string())
}

fn persist_language_preference(language: &str) -> Result<(), String> {
    let mut persisted = load_desktop_settings();
    persisted.language = language.to_string();
    persist_desktop_settings(&persisted)
}

const CANVAS: u32 = 0xF9FAFB;
const SIDEBAR: u32 = 0xFFFFFF;
const SURFACE: u32 = 0xFFFFFF;
const TEXT: u32 = 0x1F2937;
const SECONDARY: u32 = 0x475569;
const MUTED: u32 = 0x94A3B8;
const BORDER: u32 = 0xE2E8F0;
const LIME: u32 = 0x2563EB;
const LIME_PALE: u32 = 0xEFF6FF;
const OLIVE: u32 = 0x1D4ED8;
const WHITE: u32 = 0xFFFFFF;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Page {
    Overview,
    Models,
    ModelConfiguration,
    Performance,
    Api,
    Settings,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelTab {
    Installed,
    Discover,
    Downloads,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ModelConfigTab {
    Basic,
    Advanced,
}

#[derive(Clone)]
struct MtpDraftOption {
    title: String,
    path: String,
}

#[derive(Clone)]
struct BenchmarkModelOption {
    title: String,
    path: String,
}

impl DropdownItem for BenchmarkModelOption {
    type Value = String;

    fn title(&self) -> SharedString {
        SharedString::from(self.title.clone())
    }

    fn value(&self) -> &Self::Value {
        &self.path
    }
}

impl DropdownItem for MtpDraftOption {
    type Value = String;

    fn title(&self) -> SharedString {
        SharedString::from(self.title.clone())
    }

    fn value(&self) -> &Self::Value {
        &self.path
    }
}

impl Page {
    fn title(self, language: &str) -> String {
        let english = match self {
            Self::Overview => "Overview",
            Self::Models => "Models",
            Self::ModelConfiguration => "Model configuration",
            Self::Performance => "Performance Tests",
            Self::Api => "API Server",
            Self::Settings => "Settings",
        };
        tr(language, english)
    }

    fn icon_path(self) -> &'static str {
        match self {
            Self::Overview => "icons/layout-dashboard.svg",
            Self::Models => "icons/box.svg",
            Self::ModelConfiguration => "icons/settings.svg",
            Self::Performance => "icons/blocks.svg",
            Self::Api => "icons/network.svg",
            Self::Settings => "icons/settings.svg",
        }
    }
}

struct MiniMlxApp {
    page: Page,
    model_tab: ModelTab,
    configured_model: String,
    model_directory: String,
    machine_label: String,
    models: Vec<ModelEntry>,
    incomplete_downloads: Vec<IncompleteDownload>,
    selected_model: String,
    selected_library_model: Option<String>,
    model_library_filter: String,
    model_library_sort_desc: bool,
    loaded_model: Option<String>,
    backend: Arc<Mutex<Backend>>,
    operation: String,
    download_progress: Option<DownloadProgress>,
    downloading_repo: Option<String>,
    download_pause: Option<Arc<AtomicBool>>,
    report: String,
    report_count: usize,
    benchmark_running: bool,
    benchmark_error: Option<String>,
    performance_contexts: Vec<usize>,
    performance_warmup: String,
    performance_repeats: String,
    performance_max_tokens: String,
    performance_prefill_step: String,
    performance_ignore_eos: bool,
    performance_report_tab: usize,
    api_running: bool,
    api_metrics: Value,
    memory_used: u64,
    memory_total: u64,
    focus_handle: FocusHandle,
    settings_text_inputs: HashMap<String, Entity<InputState>>,
    _input_subscriptions: Vec<Subscription>,
    log_level_dropdown: Entity<DropdownState<Vec<String>>>,
    model_search_sort_dropdown: Entity<DropdownState<Vec<String>>>,
    model_config_mtp_dropdown: Entity<DropdownState<Vec<MtpDraftOption>>>,
    model_config_top_p_slider: Entity<SliderState>,
    benchmark_model_dropdown: Entity<DropdownState<Vec<BenchmarkModelOption>>>,
    benchmark_dropdown_paths: Vec<String>,
    model_search_sort: ModelSearchSort,
    repo_id: String,
    model_search_results: Vec<ModelSearchResult>,
    model_search_scroll: UniformListScrollHandle,
    selected_hub_model: Option<String>,
    model_search_error: Option<String>,
    model_search_loading: bool,
    model_search_revision: Arc<AtomicU64>,
    hub: &'static str,
    settings_category: SettingsCategory,
    settings_search: String,
    settings_port: String,
    settings_max_tokens: String,
    settings_dirty: bool,
    settings_message: String,
    settings: DesktopSettings,
    settings_context_window: String,
    settings_temperature: String,
    settings_top_p: String,
    settings_top_k: String,
    settings_repetition_penalty: String,
    settings_new_model_directory: String,
    settings_data_path: String,
    settings_huggingface_endpoint: String,
    settings_http_proxy: String,
    settings_https_proxy: String,
    settings_no_proxy: String,
    settings_ca_bundle: String,
    model_config_path: Option<String>,
    model_config_name: String,
    model_config_tab: ModelConfigTab,
    model_config_alias: String,
    model_config_max_tokens: String,
    model_config_context_tokens: String,
    model_config_temperature: String,
    model_config_top_p: String,
    model_config_top_k: String,
    model_config_repetition_penalty: String,
    model_config_min_p: String,
    model_config_presence_penalty: String,
    model_config_frequency_penalty: String,
    model_config_force_sampling: bool,
    model_config_chat_template_kwargs: String,
    model_config_mtp_draft: String,
    api_key_ready: bool,
}

#[derive(Default)]
struct ContextBenchmarkRow {
    context: String,
    ttft: String,
    tpot: String,
    prompt_tps: String,
    generation_tps: String,
    e2e_seconds: String,
    peak_memory: String,
}

fn benchmark_log(report: &str) -> String {
    if let Some((_, fenced)) = report.split_once("```text\n") {
        return fenced
            .split("\n```")
            .next()
            .unwrap_or(fenced)
            .trim()
            .to_string();
    }
    if let Some((_, output)) = report.split_once("\n\n") {
        return output.trim().to_string();
    }
    report.trim().to_string()
}

fn benchmark_speculative_draft(report: &str) -> Option<(String, String)> {
    let mut disabled = None;
    for (label, method) in [("- DFlash draft: `", "DFlash"), ("- MTP draft: `", "MTP")] {
        if let Some(draft) = report.lines().find_map(|line| {
            line.trim().strip_prefix(label).and_then(|value| value.strip_suffix('`'))
        }) {
            if draft == "disabled" {
                disabled.get_or_insert_with(|| (method.to_string(), draft.to_string()));
                continue;
            }
            return Some((method.to_string(), draft.to_string()));
        }
    }
    disabled
}

fn format_context_label(context: &str) -> String {
    context
        .parse::<usize>()
        .map(|tokens| {
            if tokens >= 1024 && tokens % 1024 == 0 {
                format!("{}K", tokens / 1024)
            } else {
                tokens.to_string()
            }
        })
        .unwrap_or_else(|_| context.to_string())
}

fn context_benchmark_rows(report: &str) -> Vec<ContextBenchmarkRow> {
    let log = benchmark_log(report);
    let mut rows: Vec<ContextBenchmarkRow> = Vec::new();
    for line in log.lines() {
        if let Some(context) = line.strip_prefix("context=") {
            let fields: HashMap<&str, &str> = line
                .split_whitespace()
                .filter_map(|part| part.split_once('='))
                .collect();
            let value = |key: &str| fields.get(key).copied().unwrap_or("—");
            let e2e_ms = value("E2E").trim_end_matches("ms").parse::<f64>().ok();
            rows.push(ContextBenchmarkRow {
                context: context.split_whitespace().next().unwrap_or("—").to_string(),
                ttft: value("TTFT").trim_end_matches("ms").to_string(),
                tpot: value("TPOT").trim_end_matches("ms").to_string(),
                prompt_tps: value("PP_TPS").to_string(),
                generation_tps: value("TG_TPS").to_string(),
                e2e_seconds: e2e_ms
                    .map(|ms| format!("{:.1}", ms / 1000.0))
                    .unwrap_or_else(|| "—".into()),
                peak_memory: "—".into(),
            });
        } else if line.trim_start().starts_with("memory ") {
            if let Some(row) = rows.last_mut() {
                row.peak_memory = line
                    .split_whitespace()
                    .find_map(|part| part.strip_prefix("peak="))
                    .map(|value| format!("{value} GiB"))
                    .unwrap_or_else(|| "—".into());
            }
        }
    }
    rows
}

fn context_benchmark_csv(report: &str) -> Option<String> {
    let rows = context_benchmark_rows(report);
    if rows.is_empty() {
        return None;
    }
    let mut csv = String::from(
        "Context,TTFT (ms),TPOT (ms),Prompt TPS (tok/s),Generation TPS (tok/s),E2E (s),Peak memory\n",
    );
    for row in rows {
        let fields = [
            format_context_label(&row.context),
            row.ttft,
            row.tpot,
            row.prompt_tps,
            row.generation_tps,
            row.e2e_seconds,
            row.peak_memory,
        ];
        csv.push_str(
            &fields
                .iter()
                .map(|field| {
                    if field
                        .chars()
                        .any(|character| matches!(character, ',' | '"' | '\n' | '\r'))
                    {
                        format!("\"{}\"", field.replace('"', "\"\""))
                    } else {
                        field.clone()
                    }
                })
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');
    }
    Some(csv)
}

fn benchmark_param_field(
    label: String,
    hint: String,
    control: impl IntoElement,
) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .flex_1()
        .min_w_0()
        .gap_4()
        .child(
            div()
                .flex_1()
                .min_w_0()
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .text_size(px(12.0))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(rgb(SECONDARY))
                        .child(label),
                )
                .child(
                    div()
                        .text_size(px(11.0))
                        .text_color(rgb(MUTED))
                        .when(!hint.is_empty(), |element| element.child(hint)),
                ),
        )
        .child(div().flex_1().min_w_0().child(control))
}

#[derive(Clone, Copy, Default)]
struct DownloadProgress {
    downloaded: u64,
    total: Option<u64>,
    bytes_per_second: f64,
}

#[derive(Action, Clone, PartialEq, Eq, Deserialize)]
#[action(namespace = model_card, no_json)]
struct CopyModelCardText {
    text: String,
}

struct ModelCardView {
    language: String,
    content: Option<String>,
    error: Option<String>,
    focus_handle: FocusHandle,
}

impl Render for ModelCardView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let copy_action_context = self.focus_handle.clone();
        let body: AnyElement = if let Some(markdown) = self.content.as_deref() {
            if markdown.trim().is_empty() {
                div()
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child("Model card is empty.")
                    .into_any_element()
            } else {
                let copy_text = markdown.to_string();
                div()
                    .id("model-card-markdown-context")
                    .w_full()
                    .context_menu(move |menu, _, _| {
                        menu.action_context(copy_action_context.clone()).menu(
                            "Copy",
                            Box::new(CopyModelCardText {
                                text: copy_text.clone(),
                            }),
                        )
                    })
                    .child(
                        TextView::markdown("model-card-markdown", markdown.to_string(), window, cx)
                            .selectable(),
                    )
                    .into_any_element()
            }
        } else if let Some(error) = self.error.as_deref() {
            div()
                .text_size(px(13.0))
                .text_color(rgb(SECONDARY))
                .child(error.to_string())
                .into_any_element()
        } else {
            div()
                .text_size(px(13.0))
                .text_color(rgb(SECONDARY))
                .child(tr(&self.language, "Loading…"))
                .into_any_element()
        };

        div()
            .track_focus(&self.focus_handle)
            .on_action(|action: &CopyModelCardText, _, cx| {
                cx.write_to_clipboard(ClipboardItem::new_string(action.text.clone()));
            })
            .w_full()
            .min_h(px(280.0))
            .text_color(rgb(TEXT))
            .child(body)
    }
}

impl MiniMlxApp {
    fn settings_input_values(&self) -> Vec<(String, String)> {
        vec![
            ("settings-search".into(), self.settings_search.clone()),
            ("benchmark-warmup".into(), self.performance_warmup.clone()),
            ("benchmark-repeats".into(), self.performance_repeats.clone()),
            (
                "benchmark-max-tokens".into(),
                self.performance_max_tokens.clone(),
            ),
            (
                "benchmark-prefill-step".into(),
                self.performance_prefill_step.clone(),
            ),
            ("model-repo-input".into(), self.repo_id.clone()),
            ("model-library-filter".into(), self.model_library_filter.clone()),
            ("setting-value-Port".into(), self.settings_port.clone()),
            (
                "setting-value-Maximum output tokens".into(),
                self.settings_max_tokens.clone(),
            ),
            (
                "setting-value-Context window".into(),
                self.settings_context_window.clone(),
            ),
            (
                "setting-value-Temperature".into(),
                self.settings_temperature.clone(),
            ),
            ("setting-value-Top P".into(), self.settings_top_p.clone()),
            ("setting-value-Top K".into(), self.settings_top_k.clone()),
            (
                "setting-value-Repetition penalty".into(),
                self.settings_repetition_penalty.clone(),
            ),
            (
                "setting-huggingface-endpoint".into(),
                self.settings_huggingface_endpoint.clone(),
            ),
            (
                "setting-download-HTTP proxy".into(),
                self.settings_http_proxy.clone(),
            ),
            (
                "setting-download-HTTPS proxy".into(),
                self.settings_https_proxy.clone(),
            ),
            (
                "setting-download-No proxy".into(),
                self.settings_no_proxy.clone(),
            ),
            (
                "setting-download-CA bundle".into(),
                self.settings_ca_bundle.clone(),
            ),
            (
                "model-directory-input".into(),
                self.settings_new_model_directory.clone(),
            ),
            ("settings-data-path".into(), self.settings_data_path.clone()),
            ("model-config-alias".into(), self.model_config_alias.clone()),
            (
                "model-config-max-tokens".into(),
                self.model_config_max_tokens.clone(),
            ),
            (
                "model-config-context-tokens".into(),
                self.model_config_context_tokens.clone(),
            ),
            (
                "model-config-temperature".into(),
                self.model_config_temperature.clone(),
            ),
            ("model-config-top-p".into(), self.model_config_top_p.clone()),
            ("model-config-top-k".into(), self.model_config_top_k.clone()),
            (
                "model-config-repetition-penalty".into(),
                self.model_config_repetition_penalty.clone(),
            ),
            ("model-config-min-p".into(), self.model_config_min_p.clone()),
            (
                "model-config-presence-penalty".into(),
                self.model_config_presence_penalty.clone(),
            ),
            (
                "model-config-frequency-penalty".into(),
                self.model_config_frequency_penalty.clone(),
            ),
            (
                "model-config-chat-template-kwargs".into(),
                self.model_config_chat_template_kwargs.clone(),
            ),
        ]
    }

    fn set_settings_input_value(&mut self, id: &str, value: &str) {
        match id {
            "settings-search" => self.settings_search = value.to_string(),
            "benchmark-warmup" => self.performance_warmup = value.to_string(),
            "benchmark-repeats" => self.performance_repeats = value.to_string(),
            "benchmark-max-tokens" => self.performance_max_tokens = value.to_string(),
            "benchmark-prefill-step" => self.performance_prefill_step = value.to_string(),
            "model-repo-input" => self.repo_id = value.to_string(),
            "model-library-filter" => self.model_library_filter = value.to_string(),
            "setting-value-Port" => self.settings_port = value.to_string(),
            "setting-value-Maximum output tokens" => self.settings_max_tokens = value.to_string(),
            "setting-value-Context window" => self.settings_context_window = value.to_string(),
            "setting-value-Temperature" => self.settings_temperature = value.to_string(),
            "setting-value-Top P" => self.settings_top_p = value.to_string(),
            "setting-value-Top K" => self.settings_top_k = value.to_string(),
            "setting-value-Repetition penalty" => {
                self.settings_repetition_penalty = value.to_string()
            }
            "setting-huggingface-endpoint" => {
                self.settings_huggingface_endpoint = value.to_string()
            }
            "setting-download-HTTP proxy" => self.settings_http_proxy = value.to_string(),
            "setting-download-HTTPS proxy" => self.settings_https_proxy = value.to_string(),
            "setting-download-No proxy" => self.settings_no_proxy = value.to_string(),
            "setting-download-CA bundle" => self.settings_ca_bundle = value.to_string(),
            "model-directory-input" => self.settings_new_model_directory = value.to_string(),
            "settings-data-path" => self.settings_data_path = value.to_string(),
            "model-config-alias" => self.model_config_alias = value.to_string(),
            "model-config-max-tokens" => self.model_config_max_tokens = value.to_string(),
            "model-config-context-tokens" => self.model_config_context_tokens = value.to_string(),
            "model-config-temperature" => self.model_config_temperature = value.to_string(),
            "model-config-top-p" => self.model_config_top_p = value.to_string(),
            "model-config-top-k" => self.model_config_top_k = value.to_string(),
            "model-config-repetition-penalty" => {
                self.model_config_repetition_penalty = value.to_string()
            }
            "model-config-min-p" => self.model_config_min_p = value.to_string(),
            "model-config-presence-penalty" => {
                self.model_config_presence_penalty = value.to_string()
            }
            "model-config-frequency-penalty" => {
                self.model_config_frequency_penalty = value.to_string()
            }
            "model-config-chat-template-kwargs" => {
                self.model_config_chat_template_kwargs = value.to_string()
            }
            _ => {}
        }
    }

    fn sync_settings_text_input(
        &mut self,
        id: &str,
        value: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        self.set_settings_input_value(id, value);
        if let Some(input) = self.settings_text_inputs.get(id).cloned() {
            let value = value.to_string();
            input.update(cx, |input, cx| input.set_value(value, window, cx));
        }
    }

    fn settings_text_control(&self, id: &str, _cx: &mut Context<Self>) -> TextInput {
        let input = self.settings_text_inputs[id].clone();
        TextInput::new(&input)
    }

    fn new(
        model_dir: String,
        root: PathBuf,
        python: PathBuf,
        machine_label: String,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) -> Self {
        let configured_model = Path::new(&model_dir)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("No model configured")
            .to_string();
        let mut settings = load_desktop_settings();
        let managed_directory = managed_models_directory();
        let managed_comparable = managed_directory
            .canonicalize()
            .unwrap_or_else(|_| managed_directory.clone());
        let configured_model_path = expand_user_directory(&model_dir)
            .canonicalize()
            .unwrap_or_else(|_| expand_user_directory(&model_dir));
        let previous_model_directories = settings.model_directories.clone();
        let mut model_directories = vec![managed_directory.display().to_string()];
        for configured in std::mem::take(&mut settings.model_directories) {
            if let Ok(directory) =
                validate_external_model_directory(&expand_user_directory(&configured))
            {
                if directory != managed_comparable
                    && directory != configured_model_path
                    && !model_directories.contains(&directory.display().to_string())
                {
                    model_directories.push(directory.display().to_string());
                }
            }
        }
        let directories_changed = previous_model_directories != model_directories;
        settings.model_directories = model_directories;
        if directories_changed {
            let _ = persist_desktop_settings(&settings);
        }
        let api_key = keychain_api_key();
        let backend = Arc::new(Mutex::new(Backend::new(
            root,
            python,
            PathBuf::from(&model_dir),
        )));
        let _ = backend.lock().unwrap().configure_api(
            if settings.bind_lan {
                "0.0.0.0"
            } else {
                "127.0.0.1"
            }
            .into(),
            settings.port,
            settings.max_tokens,
            settings.max_context_tokens,
            settings.temperature,
            settings.top_p,
            settings.top_k,
            settings.repetition_penalty,
            settings.log_level.clone(),
            settings.record_usage_history,
            PathBuf::from(&settings.data_path).join("usage.sqlite3"),
            settings.auth_enabled,
            settings.auth_enabled.then_some(api_key.clone()).flatten(),
        );
        let _ = backend.lock().unwrap().configure_download(
            settings.huggingface_endpoint.clone(),
            settings.http_proxy.clone(),
            settings.https_proxy.clone(),
            settings.no_proxy.clone(),
            settings.ca_bundle.clone(),
        );
        let _ = backend.lock().unwrap().configure_storage(
            settings
                .model_directories
                .iter()
                .map(PathBuf::from)
                .collect(),
        );
        let incomplete_downloads = backend.lock().unwrap().incomplete_downloads();
        let api_key_ready = api_key.is_some();
        let startup_security_error = (settings.auth_enabled || settings.bind_lan) && !api_key_ready;
        let start_service_on_launch = settings.start_service_on_launch && !startup_security_error;
        let (memory_used, memory_total) = backend.lock().unwrap().system_memory();
        let (report_count, report) = backend.lock().unwrap().latest_report();
        let log_levels = vec![
            "DEBUG".to_string(),
            "INFO".to_string(),
            "WARNING".to_string(),
            "ERROR".to_string(),
        ];
        let selected_log_level = log_levels
            .iter()
            .position(|level| level == &settings.log_level)
            .map(IndexPath::new);
        let log_level_dropdown =
            cx.new(|cx| DropdownState::new(log_levels, selected_log_level, window, cx));
        let model_search_sort_options = ["Name", "Size", "Download count", "Likes"]
            .into_iter()
            .map(|label| tr(&settings.language, label))
            .collect::<Vec<_>>();
        let model_search_sort_dropdown = cx.new(|cx| {
            DropdownState::new(
                model_search_sort_options.clone(),
                Some(IndexPath::new(2)),
                window,
                cx,
            )
        });
        let model_config_mtp_dropdown = cx.new(|cx| {
            DropdownState::new(
                vec![MtpDraftOption {
                    title: tr(&settings.language, "Off · use standard decoding"),
                    path: String::new(),
                }],
                Some(IndexPath::new(0)),
                window,
                cx,
            )
        });
        let model_config_top_p_slider = cx.new(|_| {
            SliderState::new()
                .min(0.0)
                .max(1.0)
                .step(0.01)
                .default_value(settings.top_p)
        });
        let benchmark_model_dropdown =
            cx.new(|cx| DropdownState::new(Vec::<BenchmarkModelOption>::new(), None, window, cx));
        let mut app = Self {
            page: Page::Overview,
            model_tab: ModelTab::Installed,
            configured_model,
            model_directory: model_dir.clone(),
            machine_label,
            models: Vec::new(),
            incomplete_downloads,
            selected_model: model_dir.clone(),
            selected_library_model: Some(model_dir.clone()),
            model_library_filter: String::new(),
            model_library_sort_desc: false,
            loaded_model: None,
            backend,
            operation: "Runtime stopped".into(),
            download_progress: None,
            downloading_repo: None,
            download_pause: None,
            report,
            report_count,
            benchmark_running: false,
            benchmark_error: None,
            performance_contexts: vec![1024, 4096, 8192, 16384],
            performance_warmup: "32".into(),
            performance_repeats: "1".into(),
            performance_max_tokens: "128".into(),
            performance_prefill_step: "1024".into(),
            performance_ignore_eos: true,
            performance_report_tab: 0,
            api_running: false,
            api_metrics: Value::Null,
            memory_used,
            memory_total,
            focus_handle: cx.focus_handle(),
            settings_text_inputs: HashMap::new(),
            _input_subscriptions: Vec::new(),
            log_level_dropdown: log_level_dropdown.clone(),
            model_search_sort_dropdown: model_search_sort_dropdown.clone(),
            model_config_mtp_dropdown: model_config_mtp_dropdown.clone(),
            model_config_top_p_slider: model_config_top_p_slider.clone(),
            benchmark_model_dropdown: benchmark_model_dropdown.clone(),
            benchmark_dropdown_paths: Vec::new(),
            model_search_sort: ModelSearchSort::Downloads,
            repo_id: String::new(),
            model_search_results: Vec::new(),
            model_search_scroll: UniformListScrollHandle::default(),
            selected_hub_model: None,
            model_search_error: None,
            model_search_loading: false,
            model_search_revision: Arc::new(AtomicU64::new(0)),
            hub: "huggingface",
            settings_category: SettingsCategory::Server,
            settings_search: String::new(),
            settings_port: settings.port.to_string(),
            settings_max_tokens: settings.max_tokens.to_string(),
            settings_dirty: false,
            settings_message: if startup_security_error {
                "API service will not start: the configured API key is missing from macOS Keychain"
                    .into()
            } else {
                String::new()
            },
            settings_context_window: settings.max_context_tokens.to_string(),
            settings_temperature: settings.temperature.to_string(),
            settings_top_p: settings.top_p.to_string(),
            settings_top_k: settings.top_k.to_string(),
            settings_repetition_penalty: settings.repetition_penalty.to_string(),
            settings_new_model_directory: String::new(),
            settings_data_path: settings.data_path.clone(),
            settings_huggingface_endpoint: settings.huggingface_endpoint.clone(),
            settings_http_proxy: settings.http_proxy.clone(),
            settings_https_proxy: settings.https_proxy.clone(),
            settings_no_proxy: settings.no_proxy.clone(),
            settings_ca_bundle: settings.ca_bundle.clone(),
            model_config_path: None,
            model_config_name: String::new(),
            model_config_tab: ModelConfigTab::Basic,
            model_config_alias: String::new(),
            model_config_max_tokens: String::new(),
            model_config_context_tokens: String::new(),
            model_config_temperature: String::new(),
            model_config_top_p: String::new(),
            model_config_top_k: String::new(),
            model_config_repetition_penalty: String::new(),
            model_config_min_p: String::new(),
            model_config_presence_penalty: String::new(),
            model_config_frequency_penalty: String::new(),
            model_config_force_sampling: false,
            model_config_chat_template_kwargs: String::new(),
            model_config_mtp_draft: String::new(),
            settings,
            api_key_ready,
        };
        let parent = cx.entity().downgrade();
        app._input_subscriptions.push(cx.subscribe(
            &log_level_dropdown,
            move |_, _, event: &DropdownEvent<Vec<String>>, cx| {
                if let DropdownEvent::Confirm(Some(level)) = event {
                    let level = level.clone();
                    let parent = parent.clone();
                    cx.defer(move |cx| {
                        let _ = parent.update(cx, |this, cx| {
                            this.settings.log_level = level;
                            this.settings_dirty = true;
                            this.settings_message.clear();
                            cx.notify();
                        });
                    });
                }
            },
        ));
        let sort_name = tr(&app.settings.language, "Name");
        let sort_size = tr(&app.settings.language, "Size");
        let sort_downloads = tr(&app.settings.language, "Download count");
        let sort_likes = tr(&app.settings.language, "Likes");
        let parent = cx.entity().downgrade();
        app._input_subscriptions.push(cx.subscribe(
            &model_search_sort_dropdown,
            move |_, _, event: &DropdownEvent<Vec<String>>, cx| {
                if let DropdownEvent::Confirm(Some(label)) = event {
                    let sort = if label == &sort_name {
                        ModelSearchSort::Name
                    } else if label == &sort_size {
                        ModelSearchSort::Size
                    } else if label == &sort_likes {
                        ModelSearchSort::Likes
                    } else if label == &sort_downloads {
                        ModelSearchSort::Downloads
                    } else {
                        return;
                    };
                    let parent = parent.clone();
                    cx.defer(move |cx| {
                        let _ = parent.update(cx, |this, cx| {
                            this.model_search_sort = sort;
                            sort_model_search_results(&mut this.model_search_results, sort);
                            cx.notify();
                        });
                    });
                }
            },
        ));
        let parent = cx.entity().downgrade();
        app._input_subscriptions.push(cx.subscribe(
            &model_config_mtp_dropdown,
            move |_, _, event: &DropdownEvent<Vec<MtpDraftOption>>, cx| {
                if let DropdownEvent::Confirm(Some(path)) = event {
                    let path = path.clone();
                    let parent = parent.clone();
                    cx.defer(move |cx| {
                        let _ = parent.update(cx, |this, cx| {
                            this.model_config_mtp_draft = path;
                            cx.notify();
                        });
                    });
                }
            },
        ));
        let parent = cx.entity().downgrade();
        app._input_subscriptions.push(cx.subscribe(
            &model_config_top_p_slider,
            move |_, _, event: &SliderEvent, cx: &mut Context<MiniMlxApp>| {
                let SliderEvent::Change(value) = event;
                let value = format!("{:.2}", value.end());
                let parent = parent.clone();
                cx.defer(move |cx| {
                    let _ = parent.update(cx, |this, cx| {
                        this.model_config_top_p = value;
                        cx.notify();
                    });
                });
            },
        ));
        let parent = cx.entity().downgrade();
        app._input_subscriptions.push(cx.subscribe(
            &benchmark_model_dropdown,
            move |_, _, event: &DropdownEvent<Vec<BenchmarkModelOption>>, cx| {
                if let DropdownEvent::Confirm(Some(path)) = event {
                    let path = path.clone();
                    let parent = parent.clone();
                    cx.defer(move |cx| {
                        let _ = parent.update(cx, |this, cx| {
                            this.selected_model = path;
                            cx.notify();
                        });
                    });
                }
            },
        ));
        let input_values = app.settings_input_values();
        let input_language = app.settings.language.clone();
        let parent = cx.entity().downgrade();
        for (id, value) in input_values {
            let field_id = id.clone();
            let placeholder = match id.as_str() {
                "settings-search" => Some("Search settings"),
                "model-library-filter" => Some("Filter downloaded models…"),
                _ => None,
            };
            let input = cx.new(|cx| {
                let mut state = InputState::new(window, cx).default_value(value);
                if let Some(placeholder) = placeholder {
                    state = state.placeholder(tr(&input_language, placeholder));
                }
                state
            });
            let parent = parent.clone();
            app._input_subscriptions.push(cx.subscribe(
                &input,
                move |_, state, event: &InputEvent, cx| {
                    if !matches!(event, InputEvent::Change) {
                        return;
                    }
                    let value = state.read(cx).value();
                    let field_id = field_id.clone();
                    let parent = parent.clone();
                    cx.defer(move |cx| {
                        let _ = parent.update(cx, |this, cx| {
                            this.set_settings_input_value(&field_id, value.as_str());
                            if field_id == "model-repo-input" {
                                this.schedule_model_search(cx);
                            }
                            if field_id != "settings-search"
                                && field_id != "model-library-filter"
                                && !field_id.starts_with("model-config-")
                            {
                                this.settings_dirty = true;
                                this.settings_message.clear();
                            }
                            cx.notify();
                        });
                    });
                },
            ));
            app.settings_text_inputs.insert(id, input);
        }
        let backend = app.backend.clone();
        let entity = cx.entity().downgrade();
        if start_service_on_launch {
            let startup_backend = backend.clone();
            let startup_model = app.model_directory.clone();
            let startup_options =
                model_inference_options_for_path(&app.settings, &startup_model).cloned();
            let startup_task = cx.background_executor().spawn(async move {
                let Ok(mut backend) = startup_backend.lock() else {
                    return Err("runtime lock poisoned".to_string());
                };
                backend.start_api_with_options(&startup_model, startup_options.as_ref())
            });
            let startup_entity = entity.clone();
            cx.spawn(async move |_, cx| {
                let result = startup_task.await;
                let _ = startup_entity.update(cx, |this, cx| {
                    if let Ok(mut backend) = this.backend.lock() {
                        backend.sync_state();
                        this.models = backend.models();
                        this.loaded_model = backend.loaded_model.clone();
                        this.api_running = backend.api_running();
                        if let Some(path) = this.loaded_model.clone() {
                            this.selected_model = path.clone();
                            this.selected_library_model = Some(path);
                        }
                    }
                    this.settings_message = result
                        .map(|_| "API service started on app launch".to_string())
                        .unwrap_or_else(|error| format!("Could not start API service: {error}"));
                    cx.notify();
                });
            })
            .detach();
        }
        let initial_backend = backend.clone();
        let initial_inventory = cx.background_executor().spawn(async move {
            initial_backend
                .lock()
                .map(|backend| backend.models())
                .unwrap_or_default()
        });
        let inventory_entity = entity.clone();
        cx.spawn(async move |_, cx| {
            let models = initial_inventory.await;
            let _ = inventory_entity.update(cx, |this, cx| {
                this.models = models;
                if !this.models.iter().any(|model| {
                    same_model_path(&model.path, &this.selected_model)
                }) {
                    if let Some(model) = this.models.first() {
                        this.selected_model = model.path.clone();
                        this.selected_library_model = Some(model.path.clone());
                    }
                }
                cx.notify();
            });
        })
        .detach();
        cx.spawn(async move |_, cx| loop {
            Timer::after(Duration::from_secs(2)).await;
            let backend = backend.clone();
            let (used, total, loaded_model, api_running, api_metrics) = cx
                .background_executor()
                .spawn(async move {
                    backend
                        .lock()
                        .map(|mut backend| {
                            backend.sync_state();
                            let memory = backend.system_memory();
                            (
                                memory.0,
                                memory.1,
                                backend.loaded_model.clone(),
                                backend.api_running(),
                                backend.api_metrics().unwrap_or(Value::Null),
                            )
                        })
                        .unwrap_or_default()
                })
                .await;
            if entity
                .update(cx, |this, cx| {
                    this.memory_used = used;
                    this.memory_total = total;
                    this.loaded_model = loaded_model;
                    this.api_running = api_running;
                    this.api_metrics = api_metrics;
                    cx.notify();
                })
                .is_err()
            {
                break;
            }
        })
        .detach();
        app
    }

    fn nav_item(&self, page: Page, cx: &mut Context<Self>) -> SidebarMenuItem {
        let selected = self.page == page;
        let app = cx.entity().clone();
        let item = SidebarMenuItem::new(page.title(&self.settings.language))
            .icon(gpui_component::Icon::default().path(page.icon_path()))
            .active(selected)
            .on_click(move |_, _, cx| {
                app.update(cx, |this, cx| {
                    this.page = page;
                    cx.notify();
                });
            });
        match page {
            Page::Models if !self.models.is_empty() => item.suffix(
                div()
                    .px_2()
                    .rounded_sm()
                    .bg(rgb(0xF1F5F9))
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(self.models.len().to_string()),
            ),
            Page::Models | Page::Api => item,
            _ => item,
        }
    }

    fn sidebar(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .w(px(224.0))
            .h_full()
            .px_3()
            .pt_6()
            .pb_4()
            .bg(rgb(SIDEBAR))
            .border_r_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .px_3()
                    .mb_8()
                    .child(
                        div()
                            .id("installed-model-list")
                            .flex()
                            .items_center()
                            .gap_2()
                            .mb_4()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_center()
                                    .w(px(28.0))
                                    .h(px(28.0))
                                    .rounded_sm()
                                    .bg(rgb(LIME))
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(rgb(WHITE))
                                    .child("▦"),
                            )
                            .child(
                                div()
                                .text_size(px(18.0))
                                    .font_weight(FontWeight::BOLD)
                                    .text_color(rgb(TEXT))
                                    .child("MiniMLX"),
                            ),
                    )
                    .child(
                        div()
                            .mt_1()
                            .text_size(px(13.0))
                            .text_color(rgb(SECONDARY))
                            .child(tr(&self.settings.language, "Local model operations")),
                    ),
            )
            .child(
                div()
                    .px_3()
                    .mb_2()
                    .text_size(px(14.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(MUTED))
                    .child(tr(&self.settings.language, "WORKSPACE")),
            )
            .child(SidebarMenu::new().children([
                self.nav_item(Page::Overview, cx),
                self.nav_item(Page::Models, cx),
                self.nav_item(Page::Performance, cx),
                self.nav_item(Page::Api, cx),
            ]))
            .child(div().flex_1())
            .child(self.settings_navigation_button(cx))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .mt_5()
                    .px_3()
                    .py_3()
                    .rounded_md()
                    .bg(rgb(0xFFFFFF))
                    .border_1()
                    .border_color(rgb(BORDER))
                    .child(
                        div()
                            .text_color(rgb(if self.api_running { LIME } else { MUTED }))
                            .child("●"),
                    )
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(rgb(TEXT))
                                    .child(tr(&self.settings.language, "Local MLX runtime")),
                            )
                            .child(div().text_size(px(13.0)).text_color(rgb(MUTED)).child(
                                if self.api_running {
                                    tr(&self.settings.language, "Service running")
                                } else {
                                    tr(&self.settings.language, "Runtime ready")
                                },
                            )),
                    ),
            )
    }

    fn settings_navigation_button(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let active = self.page == Page::Settings;
        app_button("settings-navigation")
            .w_full()
            .justify_start()
            .label(tr(&self.settings.language, "Settings"))
            .icon(
                gpui_component::Icon::default()
                    .path("icons/settings.svg")
                    .text_color(rgb(if active { LIME } else { SECONDARY })),
            )
            .text()
            .when(active, |button| button.bg(rgb(LIME_PALE)))
            .on_click(cx.listener(|this, _, _, cx| {
                this.page = Page::Settings;
                cx.notify();
            }))
    }

    fn toolbar(&self) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .justify_between()
            .h(px(44.0))
            .px_6()
            .bg(rgb(SURFACE))
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .text_size(px(14.0))
                    .text_color(rgb(SECONDARY))
                    .child(div().text_color(rgb(OLIVE)).child("◉"))
                    .child("MiniMLX Studio")
                    .child(
                        div()
                            .text_size(px(14.0))
                            .text_color(rgb(MUTED))
                            .child("· Local Runtime"),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .text_size(px(13.0))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .px_2()
                            .py_1()
                            .rounded_sm()
                            .bg(rgb(0xF1F5F9))
                            .border_1()
                            .border_color(rgb(BORDER))
                            .child(div().text_color(rgb(0x10B981)).child("●"))
                            .child(tr(&self.settings.language, "Ready"))
                            .child(div().text_color(rgb(MUTED)).child("|"))
                            .child(self.machine_label.clone()),
                    )
                    .child(
                        div()
                            .text_size(px(13.0))
                            .text_color(rgb(MUTED))
                            .child("Metal API"),
                    ),
            )
    }

    fn page_heading(&self, title: &str, _subtitle: &str) -> impl IntoElement {
        div().mb_5().child(
            div()
                .text_size(px(26.0))
                .font_weight(FontWeight::BOLD)
                .text_color(rgb(TEXT))
                .child(tr(&self.settings.language, title)),
        )
    }

    fn overview_heading(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .justify_between()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_size(px(24.0))
                            .font_weight(FontWeight::BOLD)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, "Overview")),
                    )
                    .child(
                        div()
                            .text_size(px(13.0))
                            .text_color(rgb(SECONDARY))
                            .child(tr(
                                &self.settings.language,
                                "Monitor local models, unified memory, and inference throughput.",
                            )),
                    ),
            )
            .child(refresh_models_button(cx, &self.settings.language))
    }

    fn section_title(title: &str, trailing: Option<&str>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .justify_between()
            .mb_3()
            .child(
                div()
                    .text_size(px(14.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(TEXT))
                    .child(title.to_string()),
            )
            .child(
                div()
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(trailing.unwrap_or("").to_string()),
            )
    }

    fn status_strip(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let loaded_model = self.loaded_model.clone();
        let model_label = loaded_model
            .as_deref()
            .and_then(|path| Path::new(path).file_name())
            .and_then(|name| name.to_str())
            .unwrap_or("No model loaded");
        let model_label = tr(&self.settings.language, model_label);
        div()
            .flex()
            .items_center()
            .justify_between()
            .px_5()
            .py_4()
            .bg(rgb(SURFACE))
            .border_1()
            .border_color(rgb(BORDER))
            .rounded_md()
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_3()
                            .child(
                                div()
                                    .text_size(px(16.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(rgb(TEXT))
                                    .child(tr(&self.settings.language, "Inference Service")),
                            )
                            .child(
                                div()
                                    .px_2()
                                    .py_1()
                                    .rounded_sm()
                                    .bg(rgb(if self.api_running {
                                        LIME_PALE
                                    } else {
                                        0xF1F5F9
                                    }))
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(rgb(if self.api_running {
                                        OLIVE
                                    } else {
                                        SECONDARY
                                    }))
                                    .child(if self.api_running {
                                        tr(&self.settings.language, "Running")
                                    } else {
                                        tr(&self.settings.language, "Stopped")
                                    }),
                            )
                            .child(
                                div()
                                    .px_2()
                                    .py_1()
                                    .rounded_sm()
                                    .bg(rgb(0xF1F5F9))
                                    .text_size(px(14.0))
                                    .text_color(rgb(SECONDARY))
                                    .child(format!("http://127.0.0.1:{}", self.settings_port)),
                            )
                            .child(self.copy_endpoint_button(cx)),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .text_size(px(13.0))
                            .text_color(rgb(SECONDARY))
                            .child(tr(&self.settings.language, "ACTIVE MODEL"))
                            .child(div().text_color(rgb(BORDER)).child("·"))
                            .child(
                                div()
                                    .text_color(rgb(TEXT))
                                    .font_weight(FontWeight::MEDIUM)
                                    .child(model_label),
                            ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(self.api_toggle_button(cx))
                    .child(action_button(
                        &tr(&self.settings.language, "Open API Docs"),
                        false,
                        cx,
                        Page::Api,
                    )),
            )
    }

    fn copy_endpoint_button(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let endpoint = format!("http://127.0.0.1:{}", self.settings_port);
        app_button("copy-api-endpoint")
            .label(tr(&self.settings.language, "Copy"))
            .on_click(cx.listener(move |this, _, _, cx| {
                cx.write_to_clipboard(ClipboardItem::new_string(endpoint.clone()));
                this.operation = "API endpoint copied".into();
                cx.notify();
            }))
    }

    fn resource_panel(&self) -> impl IntoElement {
        let used = format_bytes(self.memory_used);
        let total = format_bytes(self.memory_total);
        let usage_percent = if self.memory_total > 0 {
            (self.memory_used as f64 / self.memory_total as f64 * 100.0).clamp(0.0, 100.0)
        } else {
            0.0
        };
        let usage_fraction = (usage_percent / 100.0) as f32;
        let runtime_memory = if self.loaded_model.is_some() {
            "Model loaded"
        } else {
            "No model loaded"
        };
        let memory_guard_status = if self.api_running {
            match self.api_metrics["memory"]["admission"].as_str().unwrap_or("checking") {
                "open" => "Memory guard: monitoring",
                "paused" => "Memory pressure: new requests paused",
                "unknown" => "Memory status unavailable",
                _ => "Memory guard: checking",
            }
        } else {
            "Memory guard: waiting"
        };
        div()
            .bg(rgb(SURFACE))
            .border_1()
            .border_color(rgb(BORDER))
            .rounded_md()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_4()
                    .h(px(44.0))
                    .border_b_1()
                    .border_color(rgb(BORDER))
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, "System Resources")),
                    )
                    .child(
                        div()
                            .text_size(px(14.0))
                            .text_color(rgb(MUTED))
                            .child(tr(&self.settings.language, "Live snapshot · macOS")),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_start()
                    .px_4()
                    .py_4()
                    .gap_5()
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .justify_between()
                                    .child(metric_label(&tr(
                                        &self.settings.language,
                                        "Memory in use",
                                    )))
                                    .child(
                                        div()
                                            .text_size(px(14.0))
                                            .text_color(rgb(SECONDARY))
                                            .child(format!("{usage_percent:.0}%")),
                                    ),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_baseline()
                                    .gap_1()
                                    .child(
                                        div()
                                            .text_size(px(20.0))
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .text_color(rgb(TEXT))
                                            .child(used),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(13.0))
                                            .text_color(rgb(MUTED))
                                            .child(format!("/ {total}")),
                                    ),
                            )
                            .child(Progress::new().value(usage_fraction * 100.)),
                    )
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .pl_5()
                            .border_l_1()
                            .border_color(rgb(BORDER))
                            .child(metric_label(&tr(&self.settings.language, "Runtime")))
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(rgb(TEXT))
                                    .child(runtime_memory),
                            )
                            .child(div().text_size(px(14.0)).text_color(rgb(MUTED)).child(
                                if self.api_running {
                                    "MiniMLX · serving requests"
                                } else {
                                    "MiniMLX · service stopped"
                                },
                            ))
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(rgb(MUTED))
                                    .child(tr(&self.settings.language, memory_guard_status)),
                            ),
                    )
                    .child(
                        div()
                            .flex_1()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .pl_5()
                            .border_l_1()
                            .border_color(rgb(BORDER))
                            .child(metric_label(&tr(
                                &self.settings.language,
                                "GPU utilization",
                            )))
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(rgb(SECONDARY))
                                    .child(tr(
                                        &self.settings.language,
                                        "Not exposed by macOS telemetry",
                                    )),
                            )
                            .child(div().text_size(px(14.0)).text_color(rgb(MUTED)).child(tr(
                                &self.settings.language,
                                "Metal device · Apple Silicon",
                            ))),
                    ),
            )
    }

    fn service_stats(&self) -> impl IntoElement {
        let scope_source = if self.api_running {
            self.api_metrics["scope"]
                .as_str()
                .unwrap_or("Current server process")
        } else {
            "Start service to collect statistics"
        };
        let scope = tr(&self.settings.language, scope_source);
        div()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .mb_2()
                    .child(
                        div()
                            .text_size(px(13.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, "Service Statistics")),
                    )
                    .child(
                        div()
                            .text_size(px(14.0))
                            .text_color(rgb(MUTED))
                            .child(scope),
                    ),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .child(stat_card(
                        &tr(&self.settings.language, "Requests"),
                        stat_count(&self.api_metrics, "requests"),
                        &tr(&self.settings.language, "Since server start"),
                    ))
                    .child(stat_card(
                        &tr(&self.settings.language, "Active"),
                        stat_count(&self.api_metrics, "active_requests"),
                        &tr(&self.settings.language, "In-flight requests"),
                    ))
                    .child(stat_card(
                        &tr(&self.settings.language, "Prompt tokens"),
                        stat_count(&self.api_metrics, "prompt_tokens"),
                        &tr(&self.settings.language, "Processed input"),
                    ))
                    .child(stat_card(
                        &tr(&self.settings.language, "Generated tokens"),
                        stat_count(&self.api_metrics, "completion_tokens"),
                        &tr(&self.settings.language, "Completion output"),
                    )),
            )
            .child(
                div()
                    .flex()
                    .gap_2()
                    .mt_2()
                    .child(stat_card(
                        &tr(&self.settings.language, "Avg. prefill"),
                        stat_rate(&self.api_metrics, "prefill_tps"),
                        &tr(&self.settings.language, "Prompt tokens / second"),
                    ))
                    .child(stat_card(
                        &tr(&self.settings.language, "Avg. generation"),
                        stat_rate(&self.api_metrics, "generation_tps"),
                        &tr(&self.settings.language, "Generated tokens / second"),
                    ))
                    .child(stat_card(
                        &tr(&self.settings.language, "Uptime"),
                        format_uptime(&self.api_metrics["uptime_seconds"]),
                        &tr(&self.settings.language, "Current server process"),
                    ))
                    .child(stat_card(
                        &tr(&self.settings.language, "Failed requests"),
                        stat_count(&self.api_metrics, "failed_requests"),
                        &tr(&self.settings.language, "Since server start"),
                    )),
            )
    }

    fn loaded_models(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .bg(rgb(SURFACE))
            .border_1()
            .border_color(rgb(BORDER))
            .rounded_md()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .px_4()
                    .h(px(44.0))
                    .border_b_1()
                    .border_color(rgb(BORDER))
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, "Loaded Models")),
                    )
                    .child(action_button(
                        &tr(&self.settings.language, "Manage models"),
                        false,
                        cx,
                        Page::Models,
                    )),
            )
            .child(self.loaded_model_table_header())
            .child(self.loaded_model_row(cx))
    }

    fn loaded_model_table_header(&self) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .gap_3()
            .h(px(34.0))
            .px_4()
            .bg(rgb(0xF8FAFC))
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex_1()
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(rgb(SECONDARY))
                    .child(tr(&self.settings.language, "Model")),
            )
            .child(
                div()
                    .w(px(104.0))
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(tr(&self.settings.language, "Format / quant")),
            )
            .child(
                div()
                    .w(px(64.0))
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(tr(&self.settings.language, "Disk")),
            )
            .child(
                div()
                    .w(px(72.0))
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(tr(&self.settings.language, "Requests")),
            )
            .child(
                div()
                    .w(px(136.0))
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(tr(&self.settings.language, "Service")),
            )
            .child(
                div()
                    .w(px(252.0))
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(tr(&self.settings.language, "Actions")),
            )
    }

    fn loaded_model_row(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let loaded = self.loaded_model.clone();
        let name = loaded
            .as_deref()
            .and_then(|path| Path::new(path).file_name())
            .and_then(|value| value.to_str())
            .map(str::to_string);
        let size = loaded
            .as_deref()
            .and_then(|path| self.models.iter().find(|entry| same_model_path(&entry.path, path)))
            .map(|entry| format_bytes(entry.size_bytes))
            .unwrap_or_else(|| "—".into());
        let format = loaded
            .as_deref()
            .and_then(|path| self.models.iter().find(|entry| same_model_path(&entry.path, path)))
            .and_then(|entry| entry.quantization.clone())
            .unwrap_or_else(|| {
                if loaded.is_some() {
                    "Not declared".into()
                } else {
                    "—".into()
                }
            });
        let requests = stat_count(&self.api_metrics, "requests");
        let status = if self.api_running {
            format!("Running · :{}", self.settings_port)
        } else if loaded.is_some() {
            "Loaded · ready".to_string()
        } else {
            "Not loaded".to_string()
        };
        div()
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .h(px(60.0))
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_size(px(13.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgb(TEXT))
                            .child(name.unwrap_or_else(|| {
                                tr(&self.settings.language, "No models loaded")
                            })),
                    )
                    .child(div().text_size(px(13.0)).text_color(rgb(MUTED)).child(
                        if loaded.is_some() {
                            tr(&self.settings.language, "Local checkpoint")
                        } else {
                            tr(&self.settings.language, "Load a model from Models")
                        },
                    )),
            )
            .child(
                div()
                    .w(px(104.0))
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(format),
            )
            .child(
                div()
                    .w(px(64.0))
                    .text_size(px(14.0))
                    .text_color(rgb(MUTED))
                    .child(size),
            )
            .child(
                div()
                    .w(px(72.0))
                    .text_size(px(14.0))
                    .text_color(rgb(SECONDARY))
                    .child(if self.api_running {
                        requests
                    } else {
                        "—".into()
                    }),
            )
            .child(
                div()
                    .w(px(136.0))
                    .text_size(px(14.0))
                    .text_color(rgb(if self.api_running { OLIVE } else { SECONDARY }))
                    .child(status),
            )
            .child(div().w(px(126.0)).flex().items_center().gap_2().when_some(
                loaded,
                |this, path| {
                    let benchmark_path = path.clone();
                    this.child(benchmark_model_button(
                        benchmark_path,
                        &self.settings.language,
                        cx,
                    ))
                    .child(self.unload_loaded_model_button(path, cx))
                },
            ))
    }

    fn unload_loaded_model_button(
        &self,
        model_path: String,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let backend = self.backend.clone();
        app_button("overview-unload-model")
            .label(tr(&self.settings.language, "Unload"))
            .danger()
            .text()
            .on_click(cx.listener(move |this, _, _, cx| {
                if this.operation.ends_with('…') {
                    return;
                }
                this.operation = "Unloading model…".into();
                cx.notify();
                let backend = backend.clone();
                let unloaded_name = Path::new(&model_path)
                    .file_name()
                    .and_then(|name| name.to_str())
                    .unwrap_or("model")
                    .to_string();
                let unloaded_name = unloaded_name.clone();
                let task = cx.background_executor().spawn(async move {
                    let Ok(mut backend) = backend.lock() else {
                        return Err("runtime lock poisoned".to_string());
                    };
                    backend.unload_model()?;
                    Ok((backend.loaded_model.clone(), backend.api_running()))
                });
                let entity = cx.entity().downgrade();
                cx.spawn(async move |_, cx| {
                    let result = task.await;
                    let _ = entity.update(cx, |this, cx| {
                        match result {
                            Ok((loaded_model, api_running)) => {
                                this.loaded_model = loaded_model;
                                this.api_running = api_running;
                                this.operation = format!("Unloaded {unloaded_name}");
                            }
                            Err(error) => this.operation = format!("Unload failed: {error}"),
                        }
                        cx.notify();
                    });
                })
                .detach();
            }))
    }

    fn activity_pane(&self) -> impl IntoElement {
        let recent = self.api_metrics["recent_requests"]
            .as_array()
            .cloned()
            .unwrap_or_default();
        div()
            .flex()
            .flex_col()
            .w(px(264.0))
            .h_full()
            .px_4()
            .pt_6()
            .bg(rgb(SURFACE))
            .border_l_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .mb_4()
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, "Recent Activity")),
                    )
                    .child(
                        div()
                            .text_size(px(14.0))
                            .text_color(rgb(if self.api_running { OLIVE } else { MUTED }))
                            .child(if self.api_running {
                                "● LIVE"
                            } else {
                                "○ IDLE"
                            }),
                    ),
            )
            .when(!self.operation.is_empty(), |element| {
                element.child(
                    div()
                        .text_size(px(14.0))
                        .text_color(rgb(MUTED))
                        .child(tr(&self.settings.language, &self.operation)),
                )
            })
            .child(
                div()
                    .mt_5()
                    .mb_2()
                    .text_size(px(13.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(SECONDARY))
                    .child(tr(&self.settings.language, "Recent requests")),
            )
            .when(recent.is_empty(), |this| {
                this.child(
                    div()
                        .mt_2()
                        .px_3()
                        .py_3()
                        .rounded_sm()
                        .bg(rgb(0xF8FAFC))
                        .text_size(px(14.0))
                        .text_color(rgb(MUTED))
                        .child(if self.api_running {
                            "No requests in this server session yet."
                        } else {
                            "Requests will appear here when the service is running."
                        }),
                )
            })
            .children(recent.iter().take(8).map(|request| {
                let status = request["status"].as_str().unwrap_or("unknown");
                let prompt = request["prompt_tokens"].as_u64().unwrap_or(0);
                let completion = request["completion_tokens"].as_u64().unwrap_or(0);
                let duration = request["duration_ms"].as_f64().unwrap_or(0.0) / 1000.0;
                let model = request["model"]
                    .as_str()
                    .and_then(|path| Path::new(path).file_name())
                    .and_then(|name| name.to_str())
                    .unwrap_or("Inference request");
                let age = request["timestamp"]
                    .as_u64()
                    .map(format_relative_time)
                    .unwrap_or_else(|| "".into());
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .py_2()
                    .pl_3()
                    .border_l_1()
                    .border_b_1()
                    .border_color(rgb(BORDER))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .child(
                                div()
                                    .text_size(px(14.0))
                                    .font_weight(FontWeight::MEDIUM)
                                    .text_color(rgb(TEXT))
                                    .child(model.to_string()),
                            )
                            .child(div().text_size(px(13.0)).text_color(rgb(MUTED)).child(age)),
                    )
                    .child(
                        div()
                            .text_size(px(13.0))
                            .text_color(rgb(if status == "failed" {
                                0xA33A32
                            } else {
                                SECONDARY
                            }))
                            .child(format!(
                                "{status} · {prompt} in / {completion} out · {duration:.1}s"
                            )),
                    )
            }))
            .child(div().flex_1())
            .child(
                div()
                    .pt_3()
                    .border_t_1()
                    .border_color(rgb(BORDER))
                    .text_size(px(14.0))
                    .text_color(rgb(MUTED))
                    .child(tr(
                        &self.settings.language,
                        "MiniMLX · updates every 2 seconds",
                    )),
            )
    }

    fn overview(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .flex_1()
            .min_w(px(650.0))
            .child(
                div()
                    .id("overview-scroll")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .overflow_y_scroll()
                    .px_6()
                    .pt_6()
                    .pb_6()
                    .gap_5()
                    .child(self.overview_heading(cx))
                    .child(self.status_strip(cx))
                    .child(self.service_stats())
                    .child(self.resource_panel())
                    .child(self.loaded_models(cx)),
            )
            .child(self.activity_pane())
    }

    fn models_page(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let tab_content = match self.model_tab {
            ModelTab::Installed => self.installed_models(cx).into_any_element(),
            ModelTab::Discover => self.discover_models(cx).into_any_element(),
            ModelTab::Downloads => self.downloads(cx).into_any_element(),
        };
        div()
            .id("models-scroll")
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .overflow_y_scroll()
            .px_6()
            .pt_3()
            .pb_3()
            .child(
                div()
                    .flex()
                    .min_h(px(46.0))
                    .items_center()
                    .justify_between()
                    .gap_4()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_4()
                            .child(
                                div()
                                    .text_size(px(20.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(rgb(TEXT))
                                    .child(tr(&self.settings.language, "Models")),
                            )
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_1()
                                    .px_1()
                                    .py_1()
                                    .bg(rgb(0xF1F5F9))
                                    .rounded_md()
                                    .child(self.model_tab_button("Downloaded", ModelTab::Installed, cx))
                                    .child(self.model_tab_button("Discover", ModelTab::Discover, cx))
                                    .child(self.model_tab_button("Downloads", ModelTab::Downloads, cx)),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .when(self.model_tab == ModelTab::Installed, |tools| {
                                tools.child(
                                    self.settings_text_control("model-library-filter", cx)
                                        .with_size(gpui_component::Size::Small)
                                        .text_size(px(13.0))
                                        .w(px(270.0))
                                        .prefix(gpui_component::Icon::new(gpui_component::IconName::Search))
                                        .cleanable(),
                                )
                            })
                            .child(refresh_models_button(cx, &self.settings.language))
                            .child(add_model_button(&self.settings.language, cx)),
                    ),
            )
            .child(div().flex().flex_col().flex_1().min_h(px(0.0)).pt_3().child(tab_content))
    }

    fn model_configuration_page(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let selected_model = self
            .model_config_path
            .as_deref()
            .and_then(|path| self.models.iter().find(|model| model.path == path));
        div()
            .id("model-configuration-page")
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .min_w(px(0.0))
            .px_6()
            .pt_5()
            .pb_3()
            .gap_4()
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        app_button("model-configuration-back")
                            .icon(gpui_component::Icon::default().path("icons/chevron-left.svg"))
                            .tooltip(tr(&self.settings.language, "Back to models"))
                            .ghost()
                            .on_click(cx.listener(|this, _, _, cx| {
                                this.model_config_path = None;
                                this.page = Page::Models;
                                this.model_tab = ModelTab::Installed;
                                cx.notify();
                            })),
                    )
                    .child(
                        div()
                            .text_size(px(26.0))
                            .font_weight(FontWeight::BOLD)
                            .text_color(rgb(TEXT))
                            .child(self.model_config_name.clone()),
                    )
                    .child(
                        app_button("model-config-copy-name")
                            .icon(gpui_component::Icon::new(gpui_component::IconName::Copy))
                            .ghost()
                            .tooltip(tr(&self.settings.language, "Copy model name"))
                            .on_click(cx.listener(|this, _, _, cx| {
                                cx.write_to_clipboard(ClipboardItem::new_string(
                                    this.model_config_name.clone(),
                                ));
                            })),
                    )
                    .child(div().flex_1()),
            )
            .when_some(selected_model, |page, model| {
                page.child(
                    div()
                        .flex()
                        .w_full()
                        .min_w(px(0.0))
                        .items_center()
                        .gap_4()
                        .px_5()
                        .py_3()
                        .bg(rgb(0xECEFF5))
                        .border_1()
                        .border_color(rgb(0xE2E7F0))
                        .rounded_md()
                        .children([
                            model_info_item(
                                &self.settings.language,
                                "Model type",
                                model.model_type.as_deref().unwrap_or("—"),
                            ),
                            model_info_item(
                                &self.settings.language,
                                "Architecture",
                                model.architecture.as_deref().unwrap_or("—"),
                            ),
                            model_info_item(
                                &self.settings.language,
                                "Quantization",
                                model.quantization.as_deref().unwrap_or("—"),
                            ),
                            model_info_item(
                                &self.settings.language,
                                "Disk usage",
                                &format_bytes(model.size_bytes),
                            ),
                        ]),
                )
            })
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_b_1()
                    .border_color(rgb(BORDER))
                    .pb_2()
                    .child(
                        TabBar::new("model-configuration-tabs")
                            .segmented()
                            .with_size(gpui_component::Size::Small)
                            .child(Tab::new(tr(&self.settings.language, "Basic")))
                            .child(Tab::new(tr(&self.settings.language, "Advanced")))
                            .selected_index(if self.model_config_tab == ModelConfigTab::Basic {
                                0
                            } else {
                                1
                            })
                            .on_click(cx.listener(|this, index, _, cx| {
                                this.model_config_tab = if *index == 0 {
                                    ModelConfigTab::Basic
                                } else {
                                    ModelConfigTab::Advanced
                                };
                                cx.notify();
                            })),
                    ),
            )
            .child(self.model_settings_content(cx))
    }

    fn model_tab_button(
        &self,
        label: &'static str,
        tab: ModelTab,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let selected = self.model_tab == tab;
        app_button(SharedString::from(format!("model-tab-{label}")))
            .label(tr(&self.settings.language, label))
            .selected(selected)
            .ghost()
            .on_click(cx.listener(move |this, _, _, cx| {
                this.model_tab = tab;
                cx.notify();
            }))
    }

    fn installed_models(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let filter = self.model_library_filter.trim().to_lowercase();
        let mut filtered_models = self.models.iter().filter(|model| {
            filter.is_empty()
                || model.name.to_lowercase().contains(&filter)
                || model.model_type.as_deref().is_some_and(|value| value.to_lowercase().contains(&filter))
                || model.architecture.as_deref().is_some_and(|value| value.to_lowercase().contains(&filter))
        }).collect::<Vec<_>>();
        filtered_models.sort_by_key(|model| model.name.to_ascii_lowercase());
        if self.model_library_sort_desc {
            filtered_models.reverse();
        }
        let selected_path = self
            .selected_library_model
            .as_deref()
            .filter(|path| filtered_models.iter().any(|model| model.path == *path))
            .or_else(|| filtered_models.first().map(|model| model.path.as_str()));
        let selected_entry = selected_path.and_then(|path| self.models.iter().find(|m| m.path == path));
        div()
            .flex()
            .flex_1()
            .min_h(px(0.0))
            .min_w(px(0.0))
            .border_1()
            .border_color(rgb(BORDER))
            .overflow_hidden()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_w(px(0.0))
                    .min_h(px(0.0))
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .h(px(34.0))
                            .gap_3()
                            .bg(rgb(0xF8FAFC))
                            .border_b_1()
                            .border_color(rgb(BORDER))
                            .child(
                                div().flex_1().child(
                                    ListItem::new("model-name-sort")
                                        .w_full()
                                        .h_full()
                                        .on_click(cx.listener(|this, _, _, cx| {
                                            this.model_library_sort_desc = !this.model_library_sort_desc;
                                            cx.notify();
                                        }))
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap_2()
                                                .text_size(px(14.0))
                                                .text_color(rgb(SECONDARY))
                                                .child(tr(&self.settings.language, "Model name and architecture"))
                                                .child(
                                                    gpui_component::Icon::new(if self.model_library_sort_desc {
                                                        gpui_component::IconName::ChevronUp
                                                    } else {
                                                        gpui_component::IconName::ChevronDown
                                                    })
                                                    .small()
                                                    .text_color(rgb(SECONDARY)),
                                                ),
                                        ),
                                ),
                            )
                            .child(div().w(px(96.0)).px_3().text_right().text_size(px(14.0)).text_color(rgb(SECONDARY)).child(tr(&self.settings.language, "Disk"))),
                    )
                    .child(
                        div()
                            .id("installed-model-list")
                            .flex()
                            .flex_col()
                            .flex_1()
                            .min_h(px(0.0))
                            .overflow_y_scroll()
                            .children(filtered_models.iter().map(|entry| {
                                self.model_row(entry, selected_path == Some(entry.path.as_str()), cx)
                            }))
                            .when(filtered_models.is_empty(), |list| {
                                list.child(
                                    div()
                                        .p_5()
                                        .flex()
                                        .flex_col()
                                        .gap_2()
                                        .text_size(px(13.0))
                                        .text_color(rgb(SECONDARY))
                                        .child(tr(&self.settings.language, if self.models.is_empty() { "No models installed" } else { "No matching installed models" }))
                                        .when(self.models.is_empty(), |empty| empty.child(tr(&self.settings.language, "Download or add a model to manage it here."))),
                                )
                            }),
                    )
                    .when(!self.operation.is_empty(), |element| {
                        element.child(
                            div()
                                .px_4()
                                .py_2()
                                .border_t_1()
                                .border_color(rgb(BORDER))
                                .text_size(px(12.0))
                                .text_color(rgb(MUTED))
                                .child(tr(&self.settings.language, &self.operation)),
                        )
                    }),
            )
            .child(self.model_inspector(selected_entry, cx))
    }

    fn model_inspector(&self, entry: Option<&ModelEntry>, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .w(px(460.0))
            .flex_shrink_0()
            .flex()
            .flex_col()
            .gap_3()
            .p_4()
            .border_l_1()
            .border_color(rgb(BORDER))
            .bg(rgb(0xF8FAFC))
            .when_some(entry, |pane, model| {
                let (publisher, model_name) = model.name.split_once('/').unwrap_or(("", model.name.as_str()));
                let is_loaded = self
                    .loaded_model
                    .as_deref()
                    .is_some_and(|path| same_model_path(path, &model.path));
                let is_draft = is_speculative_draft_entry(model);
                let model_name_lower = model.name.to_ascii_lowercase();
                pane.child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .text_size(px(12.0))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(rgb(MUTED))
                        .child(tr(&self.settings.language, "Model information"))
                        .child(
                            app_button("copy-inspected-model-name")
                                .icon(gpui_component::Icon::new(gpui_component::IconName::Copy))
                                .tooltip(tr(&self.settings.language, "Copy model name"))
                                .ghost()
                                .on_click(cx.listener({
                                    let model_name = model.name.clone();
                                    move |_, _, _, cx| {
                                        cx.write_to_clipboard(ClipboardItem::new_string(model_name.clone()));
                                    }
                                })),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_0()
                        .child(
                            div()
                                .font_family("Menlo")
                                .text_size(px(17.0))
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(rgb(TEXT))
                                .child(if publisher.is_empty() { model.name.clone() } else { publisher.to_string() }),
                        )
                        .child(
                            div()
                                .font_family("Menlo")
                                .text_size(px(17.0))
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(rgb(0x2563EB))
                                .when(!publisher.is_empty(), |line| line.child(format!("/{model_name}"))),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .flex_wrap()
                        .gap_2()
                        .when(model_name_lower.contains("instruct"), |tags| {
                            tags.child(Tag::primary().small().outline().child(tr(&self.settings.language, "Instruct")))
                        })
                        .when(model_name_lower.contains("-vl-") || model_name_lower.contains("vision"), |tags| {
                            tags.child(pastel_success_tag().child(tr(&self.settings.language, "Vision")))
                        })
                        .when(model_name_lower.contains("moe") || model_name_lower.contains("a3b"), |tags| {
                            tags.child(Tag::secondary().small().outline().child(tr(&self.settings.language, "MoE")))
                        })
                        .when(!is_draft, |tags| {
                            tags.child(Tag::secondary().small().outline().child(tr(&self.settings.language, "MLX native")))
                        })
                        .when_some(model.quantization.as_ref(), |tags, quantization| {
                            let quantization_lower = quantization.to_ascii_lowercase();
                            let label = quantization_lower
                                .split(|character: char| !character.is_ascii_digit())
                                .find(|part| !part.is_empty())
                                .map(|bits| format!("{bits}-bit {}", tr(&self.settings.language, "quantized")))
                                .unwrap_or_else(|| quantization.clone());
                            tags.child(Tag::secondary().small().child(label))
                        })
                        .when(is_speculative_draft_entry(model), |tags| {
                            tags.child(pastel_warning_tag().child(tr(&self.settings.language, if model.architecture.as_deref().is_some_and(|a| a.to_ascii_lowercase().contains("dflash")) { "DFlash draft" } else { "MTP draft" })))
                        }),
                )
                .child(self.model_inspector_actions(model, cx))
                .child(div().h(px(1.0)).bg(rgb(BORDER)).w_full())
                .child(
                    div()
                        .text_size(px(14.0))
                        .child(
                    DescriptionList::horizontal()
                        .label_width(px(96.0))
                        .columns(1)
                        .bordered(false)
                        .with_size(gpui_component::Size::Small)
                        .child(
                            tr(&self.settings.language, "Status"),
                            div().w_full().flex().justify_end().child(if is_draft {
                                pastel_warning_tag().child(tr(&self.settings.language, if model.architecture.as_deref().is_some_and(|a| a.to_ascii_lowercase().contains("dflash")) { "DFlash draft" } else { "MTP draft" }))
                            } else if is_loaded {
                                pastel_success_tag().child(tr(&self.settings.language, "Loaded"))
                            } else {
                                pastel_success_tag().child(tr(&self.settings.language, "Ready on disk (verified)"))
                            }).into_any_element(),
                            1,
                        )
                        .child(tr(&self.settings.language, "Model family"), div().w_full().text_right().child(model.model_type.clone().unwrap_or_else(|| "—".into())).into_any_element(), 1)
                        .child(tr(&self.settings.language, "Architecture"), div().w_full().text_right().font_family("Menlo").text_size(px(14.0)).child(model.architecture.clone().unwrap_or_else(|| "—".into())).into_any_element(), 1)
                        .child(tr(&self.settings.language, "Quantization"), div().w_full().text_right().child(model.quantization.clone().unwrap_or_else(|| "—".into())).into_any_element(), 1)
                        .child(tr(&self.settings.language, "Disk"), div().w_full().text_right().child(format_bytes(model.size_bytes)).into_any_element(), 1),
                        ),
                )
                .child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .child(div().text_size(px(14.0)).text_color(rgb(SECONDARY)).child(tr(&self.settings.language, "File location")))
                        .child(
                            app_button("reveal-model-in-finder")
                                .icon(gpui_component::Icon::new(gpui_component::IconName::ExternalLink))
                                .label(tr(&self.settings.language, "Show in Finder"))
                                .text_size(px(14.0))
                                .text()
                                .on_click(cx.listener({
                                    let path = model.path.clone();
                                    move |_, _, _, _| {
                                        let _ = Command::new("open").arg("-R").arg(&path).spawn();
                                    }
                                })),
                        ),
                )
                .child(
                    div()
                        .w_full()
                        .px_3()
                        .py_2()
                        .bg(rgb(0xF1F5F9))
                        .border_1()
                        .border_color(rgb(BORDER))
                        .rounded_md()
                        .font_family("Menlo")
                        .text_size(px(13.0))
                        .text_color(rgb(SECONDARY))
                        .child(model.path.clone()),
                )
            })
            .when(entry.is_none(), |pane| {
                pane.child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .text_size(px(13.0))
                        .text_color(rgb(SECONDARY))
                        .child(tr(&self.settings.language, "No model selected"))
                        .child(tr(&self.settings.language, "Select a model to inspect its details and actions.")),
                )
            })
    }

    fn model_inspector_actions(&self, entry: &ModelEntry, cx: &mut Context<Self>) -> impl IntoElement {
        let is_loaded = self
            .loaded_model
            .as_deref()
            .is_some_and(|path| same_model_path(path, &entry.path));
        let is_draft = is_speculative_draft_entry(entry);
        let path = entry.path.clone();
        let name = entry.name.clone();
        let model_options = model_inference_options_for_path(&self.settings, &entry.path).cloned();
        let backend = self.backend.clone();
        let load_button = app_button("inspector-model-load")
            .label(tr(&self.settings.language, if is_loaded { "Unload" } else if is_draft { "Draft" } else { "Load" }))
            .when(!is_loaded && !is_draft, |button| button.primary())
            .disabled(is_draft && !is_loaded)
            .on_click(cx.listener(move |this, _, _, cx| {
                if is_draft && !is_loaded || this.operation.ends_with('…') { return; }
                this.operation = if is_loaded { "Unloading model…" } else { "Loading model…" }.into();
                cx.notify();
                let backend = backend.clone();
                let path = path.clone();
                let name = name.clone();
                let model_options = model_options.clone();
                let task = cx.background_executor().spawn(async move {
                    let Ok(mut backend) = backend.lock() else { return Err("runtime lock poisoned".to_string()); };
                    let message = if is_loaded {
                        backend.unload_model().map(|_| "Model unloaded".to_string())
                    } else {
                        backend.load_model_with_options(&path, model_options.as_ref()).map(|_| format!("Loaded {name}"))
                    }?;
                    let models = backend.models();
                    let loaded_model = backend.loaded_model.clone();
                    let api_running = backend.api_running();
                    Ok((message, models, loaded_model, api_running))
                });
                let entity = cx.entity().downgrade();
                cx.spawn(async move |_, cx| {
                    let result = task.await;
                    let _ = entity.update(cx, |this, cx| {
                        match result {
                            Ok((message, models, loaded_model, api_running)) => {
                                this.operation = message;
                                this.models = models;
                                this.loaded_model = loaded_model.clone();
                                this.api_running = api_running;
                                if let Some(path) = loaded_model { this.selected_model = path; }
                            }
                            Err(error) => this.operation = format!("Operation failed: {error}"),
                        }
                        cx.notify();
                    });
                }).detach();
            }));
        let configure_path = entry.path.clone();
        let configure_name = entry.name.clone();
        let configure_button = app_button("inspector-model-configure")
            .icon(gpui_component::Icon::default().path("icons/settings.svg"))
            .label(tr(&self.settings.language, "Configure model"))
            .on_click(cx.listener(move |this, _, window, cx| {
                this.open_model_settings(&configure_path, &configure_name, window, cx);
            }));
        let delete_path = entry.path.clone();
        let delete_name = entry.name.clone();
        let delete_button = app_button("inspector-model-delete")
            .icon(gpui_component::Icon::default().path("icons/trash-2.svg"))
            .label(tr(&self.settings.language, "Delete model"))
            .custom(
                ButtonCustomVariant::new(cx)
                    .color(rgb(0xFBE8E7).into())
                    .foreground(rgb(0x7A302B).into())
                    .border(rgb(0xF1C9C6).into())
                    .hover(rgb(0xF7DAD8).into())
                    .active(rgb(0xF1C9C6).into()),
            )
            .on_click(cx.listener(move |this, _, window, cx| {
                this.confirm_delete_model(&delete_path, &delete_name, window, cx);
            }));
        div()
            .flex()
            .items_center()
            .w_full()
            .gap_2()
            .when(!is_draft, |actions| {
                actions
                    .child(load_button.flex_1())
                    .child(configure_button.flex_1())
            })
            .when(is_draft, |actions| {
                actions.child(
                    div()
                        .flex_1()
                        .text_size(px(12.0))
                        .text_color(rgb(SECONDARY))
                        .child(tr(
                            &self.settings.language,
                            "Draft checkpoints are attached from a target model's configuration.",
                        )),
                )
            })
            .child(delete_button.when(!is_draft, |button| button.flex_1()))
    }

    fn open_model_settings(
        &mut self,
        model_path: &str,
        model_name: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let draft_is_for_same_model = self.model_config_path.as_deref() == Some(model_path);
        if !draft_is_for_same_model {
            let options = model_inference_options_for_path(&self.settings, model_path)
                .cloned()
                .unwrap_or_default();
            self.model_config_mtp_draft = options.mtp_draft_model.as_ref()
                .map(|path| format!("mtp::{path}"))
                .or_else(|| options.dflash_draft_model.as_ref().map(|path| format!("dflash::{path}")))
                .unwrap_or_default();
            self.model_config_alias = options.alias.unwrap_or_default();
            self.model_config_max_tokens = options
                .max_tokens
                .map(|value| value.to_string())
                .unwrap_or_default();
            self.model_config_context_tokens = options
                .max_context_tokens
                .map(|value| value.to_string())
                .unwrap_or_default();
            self.model_config_temperature = options
                .temperature
                .map(|value| value.to_string())
                .unwrap_or_default();
            self.model_config_top_p = options
                .top_p
                .map(|value| value.to_string())
                .unwrap_or_default();
            let slider_top_p = self
                .model_config_top_p
                .parse::<f32>()
                .unwrap_or_else(|_| self.settings_top_p.parse().unwrap_or(0.95));
            self.model_config_top_p_slider.update(cx, |slider, cx| {
                slider.set_value(slider_top_p, window, cx);
            });
            self.model_config_top_k = options
                .top_k
                .map(|value| value.to_string())
                .unwrap_or_default();
            self.model_config_repetition_penalty = options
                .repetition_penalty
                .map(|value| value.to_string())
                .unwrap_or_default();
            self.model_config_min_p = options
                .min_p
                .map(|value| value.to_string())
                .unwrap_or_default();
            self.model_config_presence_penalty = options
                .presence_penalty
                .map(|value| value.to_string())
                .unwrap_or_default();
            self.model_config_frequency_penalty = options
                .frequency_penalty
                .map(|value| value.to_string())
                .unwrap_or_default();
            self.model_config_force_sampling = options.force_sampling.unwrap_or(false);
            self.model_config_chat_template_kwargs = options
                .chat_template_kwargs
                .map(|value| serde_json::to_string_pretty(&value).unwrap_or_default())
                .unwrap_or_default();
            let default_placeholders = [
                (
                    "model-config-alias",
                    model_name
                        .rsplit('/')
                        .next()
                        .unwrap_or(model_name)
                        .to_string(),
                ),
                ("model-config-max-tokens", self.settings_max_tokens.clone()),
                (
                    "model-config-context-tokens",
                    self.settings_context_window.clone(),
                ),
                (
                    "model-config-temperature",
                    self.settings_temperature.clone(),
                ),
                ("model-config-top-p", self.settings_top_p.clone()),
                ("model-config-top-k", self.settings_top_k.clone()),
                ("model-config-min-p", "0".into()),
                (
                    "model-config-repetition-penalty",
                    self.settings_repetition_penalty.clone(),
                ),
                ("model-config-presence-penalty", "0".into()),
                ("model-config-frequency-penalty", "0".into()),
                ("model-config-chat-template-kwargs", "{}".into()),
            ];
            for (id, value) in default_placeholders {
                if let Some(input) = self.settings_text_inputs.get(id).cloned() {
                    input.update(cx, |input, cx| input.set_placeholder(value, window, cx));
                }
            }
            for (id, value) in self.settings_input_values() {
                if !id.starts_with("model-config-") {
                    continue;
                }
                if let Some(input) = self.settings_text_inputs.get(&id).cloned() {
                    input.update(cx, |input, cx| input.set_value(value, window, cx));
                }
            }
        }
        self.model_config_path = Some(model_path.to_string());
        self.model_config_name = model_name.to_string();
        self.model_config_tab = ModelConfigTab::Basic;
        let draft_options = self.mtp_draft_options(model_path);
        let selected_draft = self.model_config_mtp_draft.clone();
        self.model_config_mtp_dropdown.update(cx, |dropdown, cx| {
            dropdown.set_items(draft_options, window, cx);
            dropdown.set_selected_value(&selected_draft, window, cx);
        });
        self.page = Page::ModelConfiguration;
        cx.notify();
    }

    fn mtp_draft_options(&self, target_path: &str) -> Vec<MtpDraftOption> {
        let mut options = vec![MtpDraftOption {
            title: tr(&self.settings.language, "Off · use standard decoding"),
            path: String::new(),
        }];
        options.extend(self.models.iter().filter_map(|candidate| {
            (candidate.path != target_path
                && qwen_mtp_pair_compatible(target_path, &candidate.path))
            .then(|| MtpDraftOption {
                title: format!("MTP · {}", candidate.name),
                path: format!("mtp::{}", candidate.path),
            })
        }));
        options.extend(self.models.iter().filter_map(|candidate| {
            (candidate.path != target_path
                && qwen_dflash_pair_compatible(target_path, &candidate.path))
            .then(|| MtpDraftOption {
                title: format!("DFlash · {}", candidate.name),
                path: format!("dflash::{}", candidate.path),
            })
        }));
        options
    }

    fn model_mtp_settings_section(&self) -> AnyElement {
        let language = &self.settings.language;
        let target_path = self.model_config_path.as_deref().unwrap_or_default();
        let has_compatible_draft = self.mtp_draft_options(target_path).len() > 1;
        let draft_kind = self.model_config_mtp_draft.split_once("::").map(|(kind, _)| kind);
        div()
            .flex()
            .flex_col()
            .gap_3()
            .px_5()
            .py_4()
            .bg(rgb(SURFACE))
            .border_1()
            .border_color(rgb(BORDER))
            .rounded_lg()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .gap_3()
                    .child(model_configuration_heading(
                        language,
                        "Speculative decoding",
                        "Use a compatible draft model to accelerate text generation.",
                        gpui_component::IconName::Bot,
                    ))
                    .when_some(draft_kind, |row, kind| {
                        row.child(
                            pastel_warning_tag().child(tr(
                                language,
                                if kind == "dflash" {
                                    "DFlash configured"
                                } else {
                                    "MTP configured"
                                },
                            )),
                        )
                    }),
            )
            .child(
                v_form()
                    .gap(px(8.0))
                    .child(
                        form_field()
                            .label(tr(language, "Draft model"))
                            .child(
                                Dropdown::new(&self.model_config_mtp_dropdown)
                                    .with_size(gpui_component::Size::Small)
                                    .w_full(),
                            ),
                    )
                    .into_any_element(),
            )
            .child(Alert::info(
                "model-config-draft-info",
                tr(
                    language,
                    "Text generation only. Availability depends on target and draft model compatibility.",
                ),
            ).with_size(gpui_component::Size::Small).banner())
            .when(!has_compatible_draft, |this| {
                this.child(
                    div()
                        .text_size(px(12.0))
                        .text_color(rgb(MUTED))
                        .child(tr(language, "No compatible speculative draft installed")),
                )
            })
            .into_any_element()
    }

    fn model_settings_content(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let save = app_button("model-config-save")
            .label(tr(&self.settings.language, "Save"))
            .primary()
            .on_click(cx.listener(|this, _, _window, cx| {
                let Some(path) = this.model_config_path.clone() else {
                    return;
                };
                let parsed = (|| -> Result<ModelInferenceOptions, String> {
                    let parse = |id: &str| -> Option<String> {
                        let value = match id {
                            "model-config-temperature" => {
                                this.model_config_temperature.clone()
                            }
                            "model-config-top-p" => this.model_config_top_p.clone(),
                            _ => this.settings_text_inputs[id].read(cx).value().to_string(),
                        };
                        let value = value.trim().to_string();
                        (!value.is_empty()).then_some(value)
                    };
                    let alias = parse("model-config-alias");
                    if alias.as_ref().is_some_and(|value| {
                        value.len() > 128 || value.chars().any(char::is_whitespace)
                    }) {
                        return Err(
                            "Model alias must be at most 128 characters and contain no spaces"
                                .into(),
                        );
                    }
                    let max_tokens = parse("model-config-max-tokens")
                        .map(|value| {
                            value.parse::<u32>().map_err(|_| {
                                "Maximum output tokens must be a positive integer".to_string()
                            })
                        })
                        .transpose()?;
                    let max_context_tokens = parse("model-config-context-tokens")
                        .map(|value| {
                            value.parse::<u32>().map_err(|_| {
                                "Context window must be a positive integer".to_string()
                            })
                        })
                        .transpose()?;
                    let temperature = parse("model-config-temperature")
                        .map(|value| {
                            value
                                .parse::<f32>()
                                .map_err(|_| "Temperature must be a number".to_string())
                        })
                        .transpose()?;
                    let top_p = parse("model-config-top-p")
                        .map(|value| {
                            value
                                .parse::<f32>()
                                .map_err(|_| "Top P must be a number".to_string())
                        })
                        .transpose()?;
                    let top_k = parse("model-config-top-k")
                        .map(|value| {
                            value
                                .parse::<u32>()
                                .map_err(|_| "Top K must be a non-negative integer".to_string())
                        })
                        .transpose()?;
                    let min_p = parse("model-config-min-p")
                        .map(|value| {
                            value
                                .parse::<f32>()
                                .map_err(|_| "Min P must be a number".to_string())
                        })
                        .transpose()?;
                    let repetition_penalty = parse("model-config-repetition-penalty")
                        .map(|value| {
                            value
                                .parse::<f32>()
                                .map_err(|_| "Repetition penalty must be a number".to_string())
                        })
                        .transpose()?;
                    let presence_penalty = parse("model-config-presence-penalty")
                        .map(|value| {
                            value
                                .parse::<f32>()
                                .map_err(|_| "Presence penalty must be a number".to_string())
                        })
                        .transpose()?;
                    let frequency_penalty = parse("model-config-frequency-penalty")
                        .map(|value| {
                            value
                                .parse::<f32>()
                                .map_err(|_| "Frequency penalty must be a number".to_string())
                        })
                        .transpose()?;
                    let chat_template_kwargs = parse("model-config-chat-template-kwargs")
                        .map(|value| {
                            let parsed: Value = serde_json::from_str(&value).map_err(|error| {
                                format!("Chat template parameters must be valid JSON: {error}")
                            })?;
                            parsed.as_object().cloned().ok_or_else(|| {
                                "Chat template parameters must be a JSON object".to_string()
                            })
                        })
                        .transpose()?;
                    let selected_draft = this.model_config_mtp_draft.as_str();
                    let mtp_draft_model = selected_draft.strip_prefix("mtp::")
                        .map(str::to_string)
                        .or_else(|| (!selected_draft.is_empty() && !selected_draft.contains("::"))
                            .then(|| selected_draft.to_string()));
                    let dflash_draft_model = selected_draft.strip_prefix("dflash::").map(str::to_string);
                    if mtp_draft_model.as_ref().is_some_and(|draft| {
                        !this.models.iter().any(|candidate| {
                            candidate.path == *draft
                                && qwen_mtp_pair_compatible(&path, &candidate.path)
                        })
                    }) {
                        return Err(
                            "The selected MTP draft is no longer installed or compatible".into(),
                        );
                    }
                    if dflash_draft_model.as_ref().is_some_and(|draft| {
                        !this.models.iter().any(|candidate| {
                            candidate.path == *draft
                                && qwen_dflash_pair_compatible(&path, &candidate.path)
                        })
                    }) {
                        return Err("The selected DFlash draft is no longer installed or compatible".into());
                    }
                    if max_tokens.is_some_and(|value| value == 0 || value > 1_000_000)
                        || max_context_tokens.is_some_and(|value| value == 0 || value > 1_000_000)
                    {
                        return Err("Token limits must be between 1 and 1,000,000".into());
                    }
                    if temperature
                        .is_some_and(|value| !valid_temperature(value))
                    {
                        return Err("Temperature must be a finite, non-negative number".into());
                    }
                    if top_p
                        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
                    {
                        return Err("Top P must be between 0 and 1".into());
                    }
                    if repetition_penalty.is_some_and(|value| !value.is_finite() || value <= 0.0) {
                        return Err("Repetition penalty must be greater than 0".into());
                    }
                    if min_p
                        .is_some_and(|value| !value.is_finite() || !(0.0..=1.0).contains(&value))
                    {
                        return Err("Min P must be between 0 and 1".into());
                    }
                    if presence_penalty
                        .is_some_and(|value| !value.is_finite() || !(-2.0..=2.0).contains(&value))
                        || frequency_penalty.is_some_and(|value| {
                            !value.is_finite() || !(-2.0..=2.0).contains(&value)
                        })
                    {
                        return Err(
                            "Presence and frequency penalties must be between -2 and 2".into()
                        );
                    }
                    Ok(ModelInferenceOptions {
                        alias,
                        max_tokens,
                        max_context_tokens,
                        temperature,
                        top_p,
                        top_k,
                        min_p,
                        repetition_penalty,
                        presence_penalty,
                        frequency_penalty,
                        force_sampling: this.model_config_force_sampling.then_some(true),
                        chat_template_kwargs,
                        mtp_draft_model,
                        dflash_draft_model,
                    })
                })();
                let options = match parsed {
                    Ok(options) => options,
                    Err(error) => {
                        this.operation = error;
                        cx.notify();
                        return;
                    }
                };
                let mut updated = this.settings.clone();
                updated
                    .model_inference
                    .retain(|configured_path, _| !same_model_path(configured_path, &path));
                if options != ModelInferenceOptions::default() {
                    updated
                        .model_inference
                        .insert(normalized_model_settings_key(&path), options.clone());
                }
                if let Err(error) = persist_desktop_settings(&updated) {
                    this.operation = format!("Could not save model settings: {error}");
                    cx.notify();
                    return;
                }
                this.settings = updated;
                let restart =
                    this.api_running
                        && this
                            .loaded_model
                            .as_deref()
                            .is_some_and(|loaded| same_model_path(loaded, &path));
                let backend = this.backend.clone();
                let entity = cx.entity().downgrade();
                if restart {
                    let restart_options = options.clone();
                    let task = cx.background_executor().spawn(async move {
                        let mut backend = backend
                            .lock()
                            .map_err(|_| "runtime lock poisoned".to_string())?;
                        backend.stop_api();
                        backend.start_api_with_options(&path, Some(&restart_options))
                    });
                    cx.spawn(async move |_, cx| {
                        let result = task.await;
                        let _ = entity.update(cx, |this, cx| {
                            this.operation = result
                                .map(|_| model_settings_saved_message(&options))
                                .unwrap_or_else(|error| {
                                    format!(
                                        "Settings saved, but API service restart failed: {error}"
                                    )
                                });
                            this.api_running = this
                                .backend
                                .lock()
                                .ok()
                                .is_some_and(|mut backend| backend.api_running());
                            cx.notify();
                        });
                    })
                    .detach();
                } else {
                    this.operation = model_settings_saved_message(&options);
                    cx.notify();
                }
            }));
        let cancel = app_button("model-config-cancel")
            .label(tr(&self.settings.language, "Cancel"))
            .ghost()
            .on_click(cx.listener(|this, _, _window, cx| {
                this.model_config_path = None;
                this.page = Page::Models;
                this.model_tab = ModelTab::Installed;
                cx.notify();
            }));
        let reset = app_button("model-config-reset")
            .label(tr(&self.settings.language, "Restore defaults"))
            .ghost()
            .on_click(cx.listener(|this, _, window, cx| {
                for id in [
                    "model-config-max-tokens",
                    "model-config-alias",
                    "model-config-context-tokens",
                    "model-config-temperature",
                    "model-config-top-p",
                    "model-config-top-k",
                    "model-config-repetition-penalty",
                    "model-config-min-p",
                    "model-config-presence-penalty",
                    "model-config-frequency-penalty",
                    "model-config-chat-template-kwargs",
                ] {
                    if let Some(input) = this.settings_text_inputs.get(id).cloned() {
                        input.update(cx, |input, cx| input.set_value(String::new(), window, cx));
                    }
                }
                this.model_config_max_tokens.clear();
                this.model_config_alias.clear();
                this.model_config_context_tokens.clear();
                this.model_config_temperature.clear();
                this.model_config_top_p.clear();
                let default_top_p = this.settings_top_p.parse::<f32>().unwrap_or(0.95);
                this.model_config_top_p_slider.update(cx, |slider, cx| {
                    slider.set_value(default_top_p, window, cx);
                });
                this.model_config_top_k.clear();
                this.model_config_repetition_penalty.clear();
                this.model_config_min_p.clear();
                this.model_config_presence_penalty.clear();
                this.model_config_frequency_penalty.clear();
                this.model_config_force_sampling = false;
                this.model_config_chat_template_kwargs.clear();
                this.model_config_mtp_draft.clear();
                this.model_config_mtp_dropdown.update(cx, |dropdown, cx| {
                    dropdown.set_selected_value(&String::new(), window, cx);
                });
                cx.notify();
            }));
        let fields = match self.model_config_tab {
            ModelConfigTab::Basic => div()
                .flex()
                .flex_col()
                .gap_4()
                .child(model_configuration_panel(vec![
                    model_configuration_heading(
                        &self.settings.language,
                        "General specification",
                        "Set the model identifier and per-request token limits.",
                        gpui_component::IconName::Settings2,
                    ),
                    div()
                        .flex()
                        .flex_col()
                        .gap_2()
                        .py_2()
                        .child(
                            div()
                                .text_size(px(13.0))
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(rgb(TEXT))
                                .child(tr(&self.settings.language, "Model alias")),
                        )
                        .child(
                            self.settings_text_control("model-config-alias", cx)
                                .with_size(gpui_component::Size::Small)
                                .text_size(px(14.0))
                                .font_family(".SystemUIFont")
                                .w_full(),
                        )
                        .into_any_element(),
                    div()
                        .flex()
                        .flex_col()
                        .gap_1()
                        .child(self.model_token_limit_field(
                            "Maximum output tokens",
                            "model-config-max-tokens",
                            "Maximum generated tokens per response.",
                            cx,
                        ))
                        .child(self.model_token_limit_field(
                            "Context window",
                            "model-config-context-tokens",
                            "KV cache maximum capacity.",
                            cx,
                        ))
                        .into_any_element(),
                ]))
                .child(self.model_mtp_settings_section())
                .child(model_configuration_panel(vec![
                    model_configuration_heading(
                        &self.settings.language,
                        "Sampling parameters",
                        "Control randomness and token selection during generation.",
                        gpui_component::IconName::ALargeSmall,
                    ),
                    div()
                        .flex()
                        .gap_5()
                        .child(div().flex_1().child(self.model_settings_row(
                            "Temperature",
                            "model-config-temperature",
                            "Higher values increase output variety.",
                            cx,
                        )))
                        .child(div().flex_1().child(self.model_sampling_slider(
                            "Top P",
                            "Cumulative probability cutoff.",
                            if self.model_config_top_p.is_empty() {
                                &self.settings_top_p
                            } else {
                                &self.model_config_top_p
                            },
                            &self.model_config_top_p_slider,
                            "0.0",
                            "1.0",
                        )))
                        .into_any_element(),
                    div()
                        .flex()
                        .gap_5()
                        .child(div().flex_1().child(self.model_settings_row(
                            "Top K",
                            "model-config-top-k",
                            "Set to 0 to disable.",
                            cx,
                        )))
                        .child(div().flex_1().child(self.model_settings_row(
                            "Min P",
                            "model-config-min-p",
                            "Set to 0 to disable.",
                            cx,
                        )))
                        .into_any_element(),
                    div()
                        .flex()
                        .gap_5()
                        .child(div().flex_1().child(self.model_settings_row(
                            "Repetition penalty",
                            "model-config-repetition-penalty",
                            "Penalize tokens already present in output.",
                            cx,
                        )))
                        .child(div().flex_1().child(self.model_settings_row(
                            "Presence penalty",
                            "model-config-presence-penalty",
                            "Encourage new topics.",
                            cx,
                        )))
                        .into_any_element(),
                ]))
                .into_any_element(),
            ModelConfigTab::Advanced => div()
                .flex()
                .flex_col()
                .gap_4()
                .child(model_configuration_panel(vec![
                    model_configuration_heading(
                        &self.settings.language,
                        "Advanced sampling",
                        "Additional controls for API and template behavior.",
                        gpui_component::IconName::Settings2,
                    ),
                    self.model_settings_row(
                        "Frequency penalty",
                        "model-config-frequency-penalty",
                        "",
                        cx,
                    )
                    .into_any_element(),
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_4()
                        .min_h(px(48.0))
                        .border_b_1()
                        .border_color(rgb(BORDER))
                        .child(
                            div()
                                .flex_1()
                                .flex()
                                .flex_col()
                                .gap_1()
                                .child(div().text_size(px(13.0)).child("Force sampling"))
                                .child(
                                    div()
                                        .text_size(px(12.0))
                                        .text_color(rgb(MUTED))
                                        .child("Ignore sampling values sent by API clients."),
                                ),
                        )
                        .child(
                            Switch::new("model-config-force-sampling")
                                .checked(self.model_config_force_sampling)
                                .on_click(cx.listener(|this, checked, _, cx| {
                                    this.model_config_force_sampling = *checked;
                                    cx.notify();
                                })),
                        )
                        .into_any_element(),
                    self.model_settings_row(
                        "Chat template parameters",
                        "model-config-chat-template-kwargs",
                        "JSON object forwarded to the model chat template.",
                        cx,
                    )
                    .into_any_element(),
                ]))
                .into_any_element(),
        };
        div()
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .child(
                div()
                    .id("model-configuration-settings-scroll")
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_h(px(0.0))
                    .overflow_y_scroll()
                    .gap_3()
                    .pb_3()
                    .child(fields),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .border_t_1()
                    .border_color(rgb(BORDER))
                    .py_3()
                    .flex_shrink_0()
                    .child(cancel)
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(rgb(MUTED))
                                    .child(tr(
                                        &self.settings.language,
                                        "Blank fields use application defaults",
                                    )),
                            )
                            .child(reset)
                            .child(save),
                    ),
            )
    }

    fn model_token_limit_field(
        &self,
        label: &'static str,
        id: &'static str,
        detail: &'static str,
        cx: &mut Context<Self>,
    ) -> AnyElement {
        let value = if id == "model-config-max-tokens" {
            &self.model_config_max_tokens
        } else {
            &self.model_config_context_tokens
        };
        let inherited_value = if id == "model-config-max-tokens" {
            &self.settings_max_tokens
        } else {
            &self.settings_context_window
        };
        let selected_preset = value
            .parse::<u32>()
            .ok()
            .or_else(|| inherited_value.parse::<u32>().ok())
            .unwrap_or_default();
        let id_prefix = if id == "model-config-max-tokens" {
            "output"
        } else {
            "context"
        };

        div()
            .flex()
            .items_center()
            .gap_4()
            .min_h(px(56.0))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .w(px(338.0))
                    .flex_shrink_0()
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, label)),
                    )
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(rgb(MUTED))
                            .child(tr(&self.settings.language, detail)),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_2()
                    .child(
                        self.settings_text_control(id, cx)
                            .with_size(gpui_component::Size::Small)
                            .text_size(px(14.0))
                            .w(px(160.0)),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_1()
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(rgb(MUTED))
                                    .child(tr(&self.settings.language, "Presets")),
                            )
                            .children([(4096, "4K"), (8192, "8K"), (16384, "16K"), (32768, "32K")].into_iter().map(|(tokens, preset)| {
                                app_button(SharedString::from(format!("model-token-preset-{id_prefix}-{tokens}")))
                                    .label(preset)
                                    .selected(selected_preset == tokens)
                                    .ghost()
                                    .on_click(cx.listener(move |this, _, window, cx| {
                                        let value = tokens.to_string();
                                        this.sync_settings_text_input(id, &value, window, cx);
                                        cx.notify();
                                    }))
                            })),
                    ),
            )
            .into_any_element()
    }

    fn model_settings_row(
        &self,
        label: &'static str,
        id: &'static str,
        detail: &'static str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .gap_4()
            .min_h(px(56.0))
            .justify_between()
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .flex_1()
                    .child(
                        div()
                            .text_size(px(14.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, label)),
                    )
                    .when(!detail.is_empty(), |row| {
                        row.child(
                            div()
                                .text_size(px(12.0))
                                .text_color(rgb(MUTED))
                                .child(tr(&self.settings.language, detail)),
                        )
                    }),
            )
            .child(
                self.settings_text_control(id, cx)
                    .with_size(gpui_component::Size::Small)
                    .text_size(px(14.0))
                    .w(px(96.0)),
            )
    }

    fn model_sampling_slider(
        &self,
        label: &'static str,
        detail: &'static str,
        value: &str,
        slider: &Entity<SliderState>,
        minimum: &'static str,
        maximum: &'static str,
    ) -> AnyElement {
        div()
            .flex()
            .flex_col()
            .justify_center()
            .gap_1()
            .min_h(px(56.0))
            .child(
                div()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .text_size(px(14.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, label)),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .flex()
                            .justify_center()
                            .font_family("Menlo")
                            .text_size(px(12.0))
                            .text_color(rgb(SECONDARY))
                            .child(value.to_string()),
                    )
                    .child(
                        div()
                            .flex_1()
                            .min_w(px(0.0))
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(rgb(MUTED))
                                    .child(minimum),
                            )
                            .child(Slider::new(slider).flex_1())
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(rgb(MUTED))
                                    .child(maximum),
                            ),
                    )
            )
            .child(
                div()
                    .text_size(px(12.0))
                    .text_color(rgb(MUTED))
                    .child(tr(&self.settings.language, detail)),
            )
            .into_any_element()
    }

    fn model_row(&self, entry: &ModelEntry, selected: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let is_draft = is_speculative_draft_entry(entry);
        let is_loaded = self
            .loaded_model
            .as_deref()
            .is_some_and(|path| same_model_path(path, &entry.path));
        let model_name_lower = entry.name.to_ascii_lowercase();
        let subtitle = if is_draft {
            if entry.architecture.as_deref().is_some_and(|a| a.to_ascii_lowercase().contains("dflash")) {
                tr(&self.settings.language, "DFlash draft")
            } else {
                tr(&self.settings.language, "MTP draft")
            }
        } else {
            [entry.model_type.as_deref(), entry.quantization.as_deref()]
                .into_iter()
                .flatten()
                .collect::<Vec<_>>()
                .join(" · ")
        };
        div()
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(ListItem::new(SharedString::from(format!("model-row-{}", entry.path)))
            .w_full()
            .h(px(58.0))
            .selected(selected)
            .on_click(cx.listener({
                let path = entry.path.clone();
                move |this, _, _, cx| {
                    this.selected_library_model = Some(path.clone());
                    cx.notify();
                }
            }))
            .child(
                div()
                    .flex()
                    .w_full()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .flex()
                            .flex_1()
                            .min_w(px(0.0))
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        div()
                                            .font_family("Menlo")
                                            .text_size(px(12.0))
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .text_color(rgb(TEXT))
                                            .text_ellipsis()
                                            .child(entry.name.clone()),
                                    )
                                    .when(is_loaded, |row| {
                                        row.child(pastel_success_tag().child(tr(&self.settings.language, "Loaded")))
                                    })
                                    .when(!is_draft && model_name_lower.contains("-vl-"), |row| {
                                        row.child(pastel_success_tag().child(tr(&self.settings.language, "Vision")))
                                    })
                                    .when(!is_draft && (model_name_lower.contains("moe") || model_name_lower.contains("a3b")), |row| {
                                        row.child(Tag::secondary().small().child(tr(&self.settings.language, "MoE")))
                                    })
                                    .when(is_draft, |row| {
                                        row.child(pastel_warning_tag().child(tr(&self.settings.language, if model_name_lower.contains("dflash") { "DFlash draft" } else { "MTP draft" })))
                                    }),
                            )
                            .child(
                                div()
                                    .text_size(px(11.0))
                                    .text_color(rgb(MUTED))
                                    .child(subtitle),
                            ),
                    )
                    .child(div().w(px(72.0)).text_right().text_size(px(12.0)).text_color(rgb(SECONDARY)).child(format_bytes(entry.size_bytes))),
            ))
    }

    #[allow(dead_code)]
    fn legacy_model_row(&self, entry: &ModelEntry, selected: bool, cx: &mut Context<Self>) -> impl IntoElement {
        let is_loaded = self
            .loaded_model
            .as_deref()
            .is_some_and(|path| same_model_path(path, &entry.path));
        let is_speculative_draft = is_speculative_draft_entry(entry);
        let draft_description = if entry
            .architecture
            .as_deref()
            .is_some_and(|architecture| architecture.to_ascii_lowercase().contains("dflash"))
        {
            "DFlash draft · attach to a target model"
        } else {
            "MTP draft · attach to a target model"
        };
        let path = entry.path.clone();
        let entry_name = entry.name.clone();
        let action_name = entry_name.clone();
        let model_options = model_inference_options_for_path(&self.settings, &entry.path).cloned();
        let backend = self.backend.clone();
        let action = app_button(SharedString::from(format!("model-action-{entry_name}")))
            .label(tr(
                &self.settings.language,
                if is_speculative_draft && !is_loaded {
                    "Draft"
                } else if is_loaded {
                    "Unload"
                } else {
                    "Load"
                },
            ))
            .when(!is_loaded && !is_speculative_draft, |button| button.primary())
            .disabled(is_speculative_draft && !is_loaded)
            .on_click(cx.listener(move |this, _, _, cx| {
                if is_speculative_draft && !is_loaded {
                    return;
                }
                if this.operation.ends_with('…') {
                    return;
                }
                this.operation = if is_loaded {
                    "Unloading model…"
                } else {
                    "Loading model…"
                }
                .into();
                cx.notify();
                let backend = backend.clone();
                let path = path.clone();
                let name = action_name.clone();
                let model_options = model_options.clone();
                let task = cx.background_executor().spawn(async move {
                    let Ok(mut backend) = backend.lock() else {
                        return Err("runtime lock poisoned".to_string());
                    };
                    let message = if is_loaded {
                        backend.unload_model().map(|_| "Model unloaded".to_string())
                    } else {
                        backend
                            .load_model_with_options(&path, model_options.as_ref())
                            .map(|_| format!("Loaded {name}"))
                    }?;
                    let models = backend.models();
                    let loaded_model = backend.loaded_model.clone();
                    let api_running = backend.api_running();
                    Ok((message, models, loaded_model, api_running))
                });
                let entity = cx.entity().downgrade();
                cx.spawn(async move |_, cx| {
                    let result = task.await;
                    let _ = entity.update(cx, |this, cx| {
                        match result {
                            Ok((message, models, loaded_model, api_running)) => {
                                this.operation = message;
                                this.models = models;
                                this.loaded_model = loaded_model.clone();
                                this.api_running = api_running;
                                if let Some(path) = loaded_model {
                                    this.selected_model = path;
                                }
                            }
                            Err(error) => this.operation = format!("Operation failed: {error}"),
                        }
                        cx.notify();
                    });
                })
                .detach();
            }));
        let configuration_path = entry.path.clone();
        let configuration_name = entry.name.clone();
        let configuration_button =
            app_button(SharedString::from(format!("model-configure-{entry_name}")))
                .icon(gpui_component::Icon::default().path("icons/settings.svg"))
                .tooltip(tr(&self.settings.language, "Configure model"))
                .compact()
                .outline()
                .on_click(cx.listener(move |this, _, window, cx| {
                    this.open_model_settings(&configuration_path, &configuration_name, window, cx);
                }));
        let delete_button = app_button(SharedString::from(format!("model-delete-{entry_name}")))
            .icon(gpui_component::Icon::default().path("icons/trash-2.svg"))
            .tooltip(tr(&self.settings.language, "Delete model"))
            .compact()
            .outline()
            .danger()
            .on_click(cx.listener({
                let model_path = entry.path.clone();
                let model_name = entry.name.clone();
                move |this, _, window, cx| {
                    this.confirm_delete_model(&model_path, &model_name, window, cx);
                }
            }));
        div()
            .id(SharedString::from(format!("model-row-{}", entry.path)))
            .on_click(cx.listener({
                let selected_path = entry.path.clone();
                move |this, _, _, cx| {
                    this.selected_library_model = Some(selected_path.clone());
                    cx.notify();
                }
            }))
            .flex()
            .items_center()
            .gap_3()
            .px_4()
            .min_h(px(56.0))
            .border_t_1()
            .border_color(rgb(BORDER))
            .when(selected, |row| row.bg(rgb(0xEFF6FF)))
            .child(
                div()
                    .flex_1()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .text_size(px(14.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(rgb(TEXT))
                    .child(entry.name.clone())
                    .child(
                        div()
                            .text_size(px(11.0))
                            .text_color(rgb(MUTED))
                            .child(if is_speculative_draft {
                                if entry.architecture.as_deref().is_some_and(|a| a.to_ascii_lowercase().contains("dflash")) {
                                            "DFlash draft".to_string()
                                } else {
                                            "MTP draft".to_string()
                                }
                            } else {
                                        entry.model_type.as_deref().unwrap_or("Model").to_string()
                            }),
                    )
                    .when(is_speculative_draft, |name| {
                        name.child(div().text_size(px(11.0)).text_color(rgb(MUTED)).child(tr(
                            &self.settings.language,
                            draft_description,
                        )))
                    }),
            )
            .child(
                div()
                    .w(px(72.0))
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(format_bytes(entry.size_bytes)),
            )
            .child(
                div()
                    .w(px(94.0))
                    .text_size(px(13.0))
                    .text_color(rgb(if is_loaded { OLIVE } else { SECONDARY }))
                    .child(tr(
                        &self.settings.language,
                        if is_loaded { "Loaded" } else { "On disk" },
                    )),
            )
            .child(
                div()
                    .w(px(166.0))
                    .flex()
                    .items_center()
                    .gap_1()
                    .child(configuration_button)
                    .child(action)
                    .child(delete_button),
            )
    }

    fn confirm_delete_model(
        &mut self,
        model_path: &str,
        model_name: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        if self.operation.ends_with('…') {
            return;
        }

        let model_path = model_path.to_string();
        let model_name = model_name.to_string();
        let entity = cx.entity().downgrade();
        let language = self.settings.language.clone();

        window.open_modal(cx, move |modal, _, _| {
            let model_name = model_name.clone();
            let entity = entity.clone();
            let model_path = model_path.clone();
            let delete_name = model_name.clone();
            modal
                .title(tr(&language, "Delete this model?"))
                .width(px(440.0))
                .show_close(false)
                .overlay_closable(false)
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_4()
                        .child(
                            div()
                                .text_size(px(14.0))
                                .text_color(rgb(TEXT))
                                .child(model_name),
                        )
                        .child(
                            div()
                                .text_size(px(13.0))
                                .text_color(rgb(MUTED))
                                .child(tr(
                                    &language,
                                    "This permanently removes the model files from disk. This action cannot be undone.",
                                )),
                        )
                        .child(
                            div()
                                .flex()
                                .justify_end()
                                .gap_2()
                                .child(
                                    app_button("cancel-delete-model")
                                        .label(tr(&language, "Cancel"))
                                        .on_click(|_, window, cx| window.close_modal(cx)),
                                )
                                .child(
                                    app_button("confirm-delete-model")
                                        .label(tr(&language, "Delete permanently"))
                                        .danger()
                                        .on_click(move |_, window, app| {
                                            let _ = entity.update(app, |this, cx| {
                                                this.delete_model_after_confirmation(
                                                    &model_path,
                                                    &delete_name,
                                                    window,
                                                    cx,
                                                );
                                            });
                                        }),
                                ),
                        ),
                )
        });
    }

    fn delete_model_after_confirmation(
        &mut self,
        model_path: &str,
        model_name: &str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        window.close_modal(cx);
        self.operation = format!("Deleting {model_name}…");
        cx.notify();

        let backend = self.backend.clone();
        let path = model_path.to_string();
        let selected_path = path.clone();
        let task = cx.background_executor().spawn(async move {
            let mut backend = backend
                .lock()
                .map_err(|_| "runtime lock poisoned".to_string())?;
            backend.delete_model(&path)
        });
        let entity = cx.entity().downgrade();
        cx.spawn(async move |_, cx| match task.await {
            Ok(models) => {
                let _ = entity.update(cx, |this, cx| {
                    this.models = models;
                    if this.selected_model == selected_path {
                        this.selected_model.clear();
                    }
                    this.operation.clear();
                    cx.notify();
                });
            }
            Err(error) => {
                let _ = entity.update(cx, |this, cx| {
                    this.operation = format!("Operation failed: {error}");
                    cx.notify();
                });
            }
        })
        .detach();
    }

    fn discover_models(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let hub_label = if self.hub == "huggingface" {
            "Hugging Face"
        } else {
            "ModelScope"
        };
        div().relative().flex().flex_col().gap_3().child(
            div()
                .bg(rgb(SURFACE))
                .border_1()
                .border_color(rgb(BORDER))
                .rounded_md()
                .p_4()
                .child(Self::section_title(
                    &tr(&self.settings.language, "Add from model hub"),
                    Some(hub_label),
                ))
                .child(self.hub_selector(cx))
                .when(self.hub == "huggingface", |element| {
                    element.child(
                        div()
                            .text_size(px(12.0))
                            .text_color(rgb(SECONDARY))
                            .mb_2()
                            .child(tr(
                                &self.settings.language,
                                "Search for DFlash drafts by model or publisher; downloads use the selected repository.",
                            )),
                    )
                })
                .child(self.repo_input(cx))
                .child(self.download_button(cx)),
        )
    }

    fn hub_selector(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .gap_2()
            .mb_2()
            .child(self.hub_button("Hugging Face", "huggingface", cx))
            .child(self.hub_button("ModelScope", "modelscope", cx))
    }

    fn hub_button(
        &self,
        label: &'static str,
        hub: &'static str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let selected = self.hub == hub;
        app_button(SharedString::from(format!("hub-{hub}")))
            .label(label)
            .selected(selected)
            .ghost()
            .on_click(cx.listener(move |this, _, _, cx| {
                this.hub = hub;
                this.schedule_model_search(cx);
                cx.notify();
            }))
    }

    fn repo_input(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut field = div()
            .flex()
            .flex_col()
            .gap_1()
            .mb_2()
            .child(
                div()
                    .text_size(px(13.0))
                    .text_color(rgb(SECONDARY))
                    .child(tr(&self.settings.language, "Repository ID")),
            )
            .child(
                self.settings_text_control("model-repo-input", cx)
                    .w_full()
                    .font_family("Menlo"),
            );
        if self.repo_id.chars().count() >= 2 {
            let results = uniform_list(
                "model-search-results",
                self.model_search_results.len(),
                cx.processor(|this, range: Range<usize>, _window, cx| {
                    let mut rows = Vec::with_capacity(range.end.saturating_sub(range.start));
                    for index in range {
                        if let Some(row) = this.model_search_result_row(index, cx) {
                            rows.push(row);
                        }
                    }
                    rows
                }),
            )
            .h(px(if self.model_search_results.is_empty() {
                0.0
            } else {
                340.0
            }))
            .track_scroll(self.model_search_scroll.clone());
            field = field.child(
                div()
                    .mt_2()
                    .bg(rgb(SURFACE))
                    .border_1()
                    .border_color(rgb(BORDER))
                    .rounded_md()
                    .overflow_hidden()
                    .child(results)
                    .when(!self.model_search_results.is_empty(), |element| {
                        element.child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .px_3()
                                .py_2()
                                .border_t_1()
                                .border_color(rgb(BORDER))
                                .child(
                                    div()
                                        .text_size(px(12.0))
                                        .text_color(rgb(SECONDARY))
                                        .child(tr(&self.settings.language, "Sort by")),
                                )
                                .child(
                                    Dropdown::new(&self.model_search_sort_dropdown).w(px(160.0)),
                                ),
                        )
                    })
                    .when(self.model_search_loading, |element| {
                        element.child(
                            div()
                                .px_3()
                                .py_2()
                                .text_size(px(13.0))
                                .text_color(rgb(SECONDARY))
                                .child(tr(&self.settings.language, "Searching…")),
                        )
                    })
                    .when(
                        !self.model_search_loading
                            && self.model_search_results.is_empty()
                            && self.model_search_error.is_none(),
                        |element| {
                            element.child(
                                div()
                                    .px_3()
                                    .py_2()
                                    .text_size(px(13.0))
                                    .text_color(rgb(SECONDARY))
                                    .child(tr(&self.settings.language, "No matching models")),
                            )
                        },
                    )
                    .when_some(self.model_search_error.as_ref(), |element, error| {
                        let error = error.clone();
                        element.child(
                            app_button("model-search-error-copy")
                                .w_full()
                                .justify_start()
                                .label(error.clone())
                                .text()
                                // Use the same tested click-to-copy pattern as the API
                                // endpoint control; no extra button is added.
                                .on_click(cx.listener(move |this, _, _, cx| {
                                    cx.write_to_clipboard(ClipboardItem::new_string(error.clone()));
                                    this.operation =
                                        tr(&this.settings.language, "Error copied to clipboard");
                                    cx.notify();
                                })),
                        )
                    }),
            );
        }
        field
    }

    fn model_search_result_row(
        &self,
        index: usize,
        cx: &mut Context<Self>,
    ) -> Option<impl IntoElement> {
        let model = self.model_search_results.get(index)?.clone();
        let repo_id = model.repo_id.clone();
        let source = self.hub;
        let selected = self.selected_hub_model.as_deref() == Some(repo_id.as_str());
        let mut details = Vec::new();
        if let Some(parameters) = model.parameters {
            details.push(format_model_parameters(parameters));
        }
        if let Some(size) = model.size_bytes {
            if size > 0 {
                details.push(format_bytes(size));
            }
        }
        let popularity = model
            .downloads
            .map(|count| format!("{} ↓", format_compact_count(count)));
        let likes = model
            .likes
            .map(|count| format!("♡ {}", format_compact_count(count)))
            .unwrap_or_default();
        let card_repo_id = repo_id.clone();
        let select_repo_id = repo_id.clone();
        Some(
            ListItem::new(SharedString::from(format!("hub-search-result-{repo_id}")))
                .w_full()
                .h(px(56.0))
                // ListItem's selected overlay is painted above its children, which
                // would intercept the nested model-card button. Keep selection as
                // ordinary component styling so child controls stay interactive.
                .bg(rgb(if selected { LIME_PALE } else { SURFACE }))
                .child(
                    div()
                        .flex()
                        .w_full()
                        .items_center()
                        .justify_between()
                        .gap_4()
                        .child(
                            div()
                                .flex()
                                .flex_1()
                                .min_w_0()
                                .items_center()
                                .gap_3()
                                .child(
                                    gpui_component::Icon::default()
                                        .path("icons/box.svg")
                                        .size(px(18.0))
                                        .text_color(rgb(MUTED)),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .flex_col()
                                        .flex_1()
                                        .min_w_0()
                                        .gap_0p5()
                                        .child(
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap_2()
                                                .child(
                                                    app_button(SharedString::from(format!(
                                                        "select-model-{repo_id}"
                                                    )))
                                                    .label(repo_id.clone())
                                                    .text()
                                                    .compact()
                                                    .flex_shrink_0()
                                                    .on_click(cx.listener(
                                                        move |this, _, window, cx| {
                                                            this.sync_settings_text_input(
                                                                "model-repo-input",
                                                                &select_repo_id,
                                                                window,
                                                                cx,
                                                            );
                                                            this.selected_hub_model =
                                                                Some(select_repo_id.clone());
                                                            this.model_search_error = None;
                                                            this.model_search_revision
                                                                .fetch_add(1, Ordering::Relaxed);
                                                            cx.notify();
                                                        },
                                                    )),
                                                )
                                                .child(
                                                    app_button(SharedString::from(format!(
                                                        "model-card-{repo_id}"
                                                    )))
                                                    .flex_shrink_0()
                                                    .label(tr(
                                                        &self.settings.language,
                                                        "Model card",
                                                    ))
                                                    .outline()
                                                    .on_click(cx.listener(
                                                        move |this, _, window, cx| {
                                                            this.open_model_card(
                                                                &card_repo_id,
                                                                source,
                                                                window,
                                                                cx,
                                                            );
                                                        },
                                                    )),
                                                ),
                                        )
                                        .child(
                                            div()
                                                .text_size(px(12.0))
                                                .text_color(rgb(MUTED))
                                                .child(details.join(" · ")),
                                        ),
                                ),
                        )
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .flex_shrink_0()
                                .items_end()
                                .gap_0p5()
                                .text_size(px(12.0))
                                .text_color(rgb(MUTED))
                                .when_some(popularity, |element, popularity| {
                                    element.child(popularity)
                                })
                                .when(!likes.is_empty(), |element| {
                                    element.child(div().text_size(px(11.0)).child(likes.clone()))
                                }),
                        ),
                ),
        )
    }

    fn open_model_card(
        &mut self,
        repo_id: &str,
        source: &'static str,
        window: &mut Window,
        cx: &mut Context<Self>,
    ) {
        let repo_id = repo_id.to_string();
        let language = self.settings.language.clone();
        let card_view = cx.new(|cx| ModelCardView {
            language,
            content: None,
            error: None,
            focus_handle: cx.focus_handle(),
        });
        let modal_card_view = card_view.clone();
        let modal_title = repo_id.clone();
        let copy_title = modal_title.clone();
        let title_focus_handle = card_view.read(cx).focus_handle.clone();
        window.open_modal(cx, move |modal, _, _| {
            let copy_title = copy_title.clone();
            let title_focus_handle = title_focus_handle.clone();
            modal
                .title(
                    div()
                        .w_full()
                        .flex()
                        .items_center()
                        .justify_between()
                        .pr(px(36.0))
                        .child(
                            div()
                                .id("model-card-title-context")
                                .context_menu(move |menu, _, _| {
                                    menu.action_context(title_focus_handle.clone()).menu(
                                        "Copy",
                                        Box::new(CopyModelCardText {
                                            text: copy_title.clone(),
                                        }),
                                    )
                                })
                                .child(modal_title.clone()),
                        )
                        .child(
                            Button::new("close-model-card")
                                .small()
                                .ghost()
                                .icon(gpui_component::Icon::default().path("icons/x.svg"))
                                .on_click(|_, window, cx| window.close_modal(cx)),
                        ),
                )
                .width(px(980.0))
                .max_w(px(1300.0))
                .min_h(px(520.0))
                .max_h(px(760.0))
                .show_close(false)
                .overlay_closable(false)
                .child(modal_card_view.clone())
        });
        let backend = self.backend.clone();
        let request_repo_id = repo_id.clone();
        let task = cx.background_executor().spawn(async move {
            let context = backend
                .lock()
                .map_err(|_| "runtime lock poisoned".to_string())?
                .download_context();
            Backend::fetch_model_card(&context, &request_repo_id, source)
        });
        let card_view = card_view.downgrade();
        cx.spawn(async move |_, cx| {
            let result = task.await;
            let _ = card_view.update(cx, |card, cx| {
                match result {
                    Ok(content) => card.content = Some(content),
                    Err(error) => card.error = Some(error),
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn schedule_model_search(&mut self, cx: &mut Context<Self>) {
        let query = self.repo_id.trim().to_string();
        let revision = self
            .model_search_revision
            .fetch_add(1, Ordering::Relaxed)
            .wrapping_add(1);
        let search_revision = self.model_search_revision.clone();
        self.model_search_results.clear();
        self.selected_hub_model = None;
        self.model_search_error = None;
        self.model_search_loading = false;
        if query.chars().count() < 2 {
            cx.notify();
            return;
        }
        self.model_search_loading = true;
        let source = self.hub;
        let backend = self.backend.clone();
        let search_query = query.clone();
        let task = cx.background_executor().spawn(async move {
            Timer::after(Duration::from_millis(450)).await;
            if search_revision.load(Ordering::Relaxed) != revision {
                return Ok((Vec::new(), None));
            }
            let context = backend
                .lock()
                .map_err(|_| "runtime lock poisoned".to_string())?
                .download_context();
            Backend::search_models(&context, &search_query, source)
                .map(|results| (results, Some(context)))
        });
        let entity = cx.entity().downgrade();
        cx.spawn(async move |_, cx| {
            let result = task.await;
            let _ = entity.update(cx, |this, cx| {
                if this.model_search_revision.load(Ordering::Relaxed) == revision
                    && this.repo_id.trim() == query
                {
                    this.model_search_loading = false;
                    match result {
                        Ok((mut results, context)) => {
                            sort_model_search_results(&mut results, this.model_search_sort);
                            this.model_search_results = results.clone();
                            if source == "huggingface" && !results.is_empty() {
                                if let Some(context) = context {
                                    let size_task = cx.background_executor().spawn(async move {
                                        let _ =
                                            Backend::populate_model_sizes(&context, &mut results);
                                        results
                                    });
                                    let size_entity = entity.clone();
                                    cx.spawn(async move |_, cx| {
                                        let sized_results = size_task.await;
                                        let _ = size_entity.update(cx, |this, cx| {
                                            if this.model_search_revision.load(Ordering::Relaxed)
                                                == revision
                                                && this.repo_id.trim() == query
                                            {
                                                for sized in sized_results {
                                                    if let Some(current) = this
                                                        .model_search_results
                                                        .iter_mut()
                                                        .find(|current| {
                                                            current.repo_id == sized.repo_id
                                                        })
                                                    {
                                                        current.size_bytes = sized.size_bytes;
                                                    }
                                                }
                                                sort_model_search_results(
                                                    &mut this.model_search_results,
                                                    this.model_search_sort,
                                                );
                                                cx.notify();
                                            }
                                        });
                                    })
                                    .detach();
                                }
                            }
                        }
                        Err(error) => this.model_search_error = Some(error),
                    }
                    cx.notify();
                }
            });
        })
        .detach();
    }

    fn download_button(&self, cx: &mut Context<Self>) -> impl IntoElement {
        app_button("download-model")
            .label(tr(&self.settings.language, "Download Model"))
            .primary()
            .mt_3()
            .on_click(cx.listener(move |this, _, _, cx| {
                if this.download_progress.is_some() {
                    return;
                }
                let repo_id = this.repo_id.trim().to_string();
                if repo_id.is_empty() || repo_id.chars().any(char::is_whitespace) {
                    this.operation = tr(
                        &this.settings.language,
                        "Enter a valid model repository ID before downloading.",
                    );
                    cx.notify();
                    return;
                }
                this.begin_model_download(repo_id, this.hub.to_string(), cx);
            }))
    }

    fn begin_model_download(&mut self, repo_id: String, source: String, cx: &mut Context<Self>) {
        self.operation = format!("Downloading {repo_id}…");
        self.model_tab = ModelTab::Downloads;
        self.downloading_repo = Some(repo_id.clone());
        self.download_progress = Some(DownloadProgress::default());
        let pause_requested = Arc::new(AtomicBool::new(false));
        self.download_pause = Some(pause_requested.clone());
        let backend = self.backend.clone();
        let shared_progress = Arc::new(Mutex::new(DownloadProgress::default()));
        let progress_for_download = shared_progress.clone();
        let finished = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let finished_for_download = finished.clone();
        let task = cx.background_executor().spawn(async move {
            let outcome = async {
                let context = backend
                    .lock()
                    .map_err(|_| "runtime lock poisoned".to_string())?
                    .download_context();
                let download_result = Backend::download_model_with_progress(
                    &context,
                    &repo_id,
                    &source,
                    |downloaded, total| {
                        if let Ok(mut progress) = progress_for_download.lock() {
                            progress.downloaded = downloaded;
                            progress.total = total;
                        }
                        !pause_requested.load(Ordering::Relaxed)
                    },
                );
                let backend_guard = backend
                    .lock()
                    .map_err(|_| "runtime lock poisoned".to_string())?;
                Ok::<_, String>((
                    download_result,
                    backend_guard.models(),
                    backend_guard.incomplete_downloads(),
                ))
            }
            .await;
            finished_for_download.store(true, Ordering::Relaxed);
            outcome
        });
        let progress_entity = cx.entity().downgrade();
        cx.spawn(async move |_, cx| {
            let mut previous_bytes = 0u64;
            let mut previous_at = std::time::Instant::now();
            loop {
                Timer::after(Duration::from_millis(400)).await;
                if finished.load(Ordering::Relaxed) {
                    break;
                }
                let progress = shared_progress.lock().ok().map(|value| *value);
                if let Some(mut progress) = progress {
                    let now = std::time::Instant::now();
                    let elapsed = now.duration_since(previous_at).as_secs_f64();
                    if elapsed > 0.0 {
                        progress.bytes_per_second =
                            progress.downloaded.saturating_sub(previous_bytes) as f64 / elapsed;
                    }
                    previous_bytes = progress.downloaded;
                    previous_at = now;
                    let _ = progress_entity.update(cx, |this, cx| {
                        this.download_progress = Some(progress);
                        cx.notify();
                    });
                }
            }
        })
        .detach();
        let entity = cx.entity().downgrade();
        cx.spawn(async move |_, cx| {
            let result = task.await;
            let _ = entity.update(cx, |this, cx| {
                this.download_progress = None;
                this.download_pause = None;
                this.downloading_repo = None;
                match result {
                    Ok((download_result, models, incomplete)) => {
                        this.models = models;
                        this.incomplete_downloads = incomplete;
                        match download_result {
                            Ok(Some(path)) => {
                                this.operation.clear();
                                this.selected_model = path.clone();
                                this.configured_model = this
                                    .models
                                    .iter()
                                    .find(|model| model.path == path)
                                    .map(|model| model.name.clone())
                                    .unwrap_or_else(|| "Model".to_string());
                            }
                            Ok(None) => {
                                this.operation.clear();
                            }
                            Err(error) => this.operation = format!("Download failed: {error}"),
                        }
                    }
                    Err(error) => this.operation = format!("Download failed: {error}"),
                }
                cx.notify();
            });
        })
        .detach();
        cx.notify();
    }

    fn downloads(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let mut content = div().flex().flex_col().gap_3().child(Self::section_title(
            &tr(&self.settings.language, "Download queue"),
            None,
        ));
        if self.download_progress.is_some() || self.operation.starts_with("Download failed:") {
            content = content.child(self.download_status(cx));
        }
        let resumable_downloads = self
            .incomplete_downloads
            .iter()
            .filter(|download| Some(download.repo_id.as_str()) != self.downloading_repo.as_deref())
            .cloned()
            .collect::<Vec<_>>();
        if resumable_downloads.is_empty() && self.download_progress.is_none() {
            content = content.child(
                div()
                    .px_4()
                    .py_4()
                    .bg(rgb(SURFACE))
                    .border_1()
                    .border_color(rgb(BORDER))
                    .rounded_md()
                    .text_size(px(14.0))
                    .text_color(rgb(MUTED))
                    .child(tr(&self.settings.language, "No incomplete downloads")),
            );
        } else if !resumable_downloads.is_empty() {
            for download in resumable_downloads {
                content = content.child(
                    div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .gap_4()
                        .px_4()
                        .py_3()
                        .bg(rgb(SURFACE))
                        .border_1()
                        .border_color(rgb(BORDER))
                        .rounded_md()
                        .child(
                            div()
                                .flex()
                                .flex_col()
                                .gap_1()
                                .child(
                                    div()
                                        .text_size(px(14.0))
                                        .font_weight(FontWeight::MEDIUM)
                                        .text_color(rgb(TEXT))
                                        .child(download.repo_id.clone()),
                                )
                                .child(div().text_size(px(12.0)).text_color(rgb(MUTED)).child(
                                    format!(
                                        "{} · {}",
                                        tr(&self.settings.language, "Incomplete download"),
                                        download
                                            .size_bytes
                                            .map(format_bytes)
                                            .unwrap_or_else(|| "—".to_string())
                                    ),
                                )),
                        )
                        .child(
                            div()
                                .flex()
                                .items_center()
                                .gap_2()
                                .when(self.download_progress.is_none(), |element| {
                                    element.child(self.resume_download_button(download.clone(), cx))
                                })
                                .child(self.delete_incomplete_download_button(download, cx)),
                        ),
                );
            }
        }
        content
    }

    fn download_status(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let message = tr(&self.settings.language, &self.operation);
        let copy_message = message.clone();
        let failed = self.operation.starts_with("Download failed:");
        let progress = self.download_progress;
        let mut status = div()
            .id("download-status")
            .flex()
            .flex_col()
            .items_center()
            .w_full()
            .min_h(px(64.0))
            .px_4()
            .py_3()
            .bg(rgb(SURFACE))
            .border_1()
            .border_color(rgb(BORDER))
            .rounded_md()
            .text_size(px(13.0))
            .text_color(rgb(MUTED))
            .child(div().w_full().whitespace_normal().child(message));
        if let Some(progress) = progress {
            let details = match progress.total {
                Some(total) if total > 0 => format!(
                    "{} / {} · {}",
                    format_bytes(progress.downloaded),
                    format_bytes(total),
                    format_download_speed(progress.bytes_per_second)
                ),
                _ => format!(
                    "{} · {}",
                    format_bytes(progress.downloaded),
                    format_download_speed(progress.bytes_per_second)
                ),
            };
            status = status.child(
                div()
                    .w_full()
                    .flex()
                    .items_center()
                    .gap_3()
                    .child(
                        div()
                            .w(px(240.0))
                            .flex_shrink_0()
                            .text_size(px(12.0))
                            .child(details),
                    )
                    .child(div().flex_1().when(
                        progress.total.is_some_and(|total| total > 0),
                        |element| {
                            element.child(Progress::new().value(
                                progress.downloaded as f32 / progress.total.unwrap_or(1) as f32
                                    * 100.0,
                            ))
                        },
                    ))
                    .child(
                        app_button("pause-model-download")
                            .icon(gpui_component::Icon::default().path("icons/pause.svg"))
                            .tooltip(tr(&self.settings.language, "Pause download"))
                            .compact()
                            .outline()
                            .disabled(
                                self.download_pause
                                    .as_ref()
                                    .is_some_and(|flag| flag.load(Ordering::Relaxed)),
                            )
                            .on_click(cx.listener({
                                let pause_flag = self.download_pause.clone();
                                move |_, _, _, cx| {
                                    if let Some(pause_flag) = &pause_flag {
                                        pause_flag.store(true, Ordering::Relaxed);
                                    }
                                    cx.notify();
                                }
                            })),
                    ),
            );
        }
        status.when(failed, |element| {
            element
                .cursor_pointer()
                .on_click(cx.listener(move |_, _, _, cx| {
                    cx.write_to_clipboard(ClipboardItem::new_string(copy_message.clone()));
                }))
        })
    }

    fn resume_download_button(
        &self,
        download: IncompleteDownload,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let busy = self.download_progress.is_some();
        app_button(SharedString::from(format!("resume-{}", download.repo_id)))
            .label(tr(&self.settings.language, "Resume"))
            .primary()
            .disabled(busy)
            .on_click(cx.listener(move |this, _, _, cx| {
                if this.download_progress.is_some() {
                    return;
                }
                let repo_id = download.repo_id.clone();
                let source = download.source.clone();
                this.model_tab = ModelTab::Downloads;
                this.begin_model_download(repo_id, source, cx);
            }))
    }

    fn delete_incomplete_download_button(
        &self,
        download: IncompleteDownload,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        app_button(SharedString::from(format!("delete-{}", download.repo_id)))
            .icon(gpui_component::Icon::default().path("icons/trash-2.svg"))
            .tooltip(tr(&self.settings.language, "Delete incomplete download"))
            .compact()
            .outline()
            .on_click(cx.listener(move |this, _, _, cx| {
                let repo_id = download.repo_id.clone();
                let backend = this.backend.clone();
                let task = cx.background_executor().spawn(async move {
                    let backend = backend
                        .lock()
                        .map_err(|_| "runtime lock poisoned".to_string())?;
                    backend.delete_incomplete_download(&repo_id)?;
                    Ok::<_, String>(backend.incomplete_downloads())
                });
                let entity = cx.entity().downgrade();
                cx.spawn(async move |_, cx| match task.await {
                    Ok(downloads) => {
                        let _ = entity.update(cx, |this, cx| {
                            this.incomplete_downloads = downloads;
                            cx.notify();
                        });
                    }
                    Err(error) => {
                        let _ = entity.update(cx, |this, cx| {
                            this.operation = format!("Download failed: {error}");
                            cx.notify();
                        });
                    }
                })
                .detach();
            }))
    }

    fn performance_page(&self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let configured_mtp_draft =
            model_inference_options_for_path(&self.settings, &self.selected_model)
                .and_then(|options| options.dflash_draft_model.as_deref().or(options.mtp_draft_model.as_deref()))
                .map(|path| {
                    Path::new(path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or(path)
                        .to_string()
                });
        div()
            .id("performance-scroll")
            .flex()
            .flex_col()
            .flex_1()
            .overflow_y_scroll()
            .px_6()
            .pt_5()
            .pb_4()
            .gap_4()
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .pb_4()
                    .border_b_1()
                    .border_color(rgb(BORDER))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_3()
                                    .child(
                                        div()
                                            .text_size(px(24.0))
                                            .font_weight(FontWeight::BOLD)
                                            .text_color(rgb(TEXT))
                                            .child(tr(&self.settings.language, "Performance Tests")),
                                    ),
                            )
                            .child(
                                div()
                                    .text_size(px(13.0))
                                    .text_color(rgb(SECONDARY))
                                    .child(tr(&self.settings.language, "Measure local model prefill throughput, decode latency, and memory usage.")),
                            ),
                    )
                    .child(self.run_benchmark_button(cx)),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_4()
                    .px_5()
                    .py_4()
                    .bg(rgb(SURFACE))
                    .border_1()
                    .border_color(rgb(BORDER))
                    .rounded_lg()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .pb_3()
                            .border_b_1()
                            .border_color(rgb(BORDER))
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_2()
                                    .child(
                                        gpui_component::Icon::default()
                                            .path("icons/settings.svg")
                                            .text_color(rgb(SECONDARY)),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(14.0))
                                            .font_weight(FontWeight::SEMIBOLD)
                                            .text_color(rgb(TEXT))
                                            .child(tr(&self.settings.language, "Test configuration")),
                                    ),
                            )
                            .child(
                                div()
                                    .text_size(px(12.0))
                                    .text_color(rgb(MUTED))
                                    .child(if let Some(draft) = configured_mtp_draft.as_ref() {
                                        format!("{} · {draft}", tr(&self.settings.language, "Speculative draft configured"))
                                    } else {
                                        tr(&self.settings.language, "Speculative decoding is not configured for this model")
                                    }),
                            ),
                    )
                    .child(benchmark_param_field(
                        tr(&self.settings.language, "Model"),
                        String::new(),
                        Dropdown::new(&self.benchmark_model_dropdown)
                            .small()
                            .placeholder(tr(&self.settings.language, "Select model"))
                            .w_full(),
                    ))
                    .child(
                        div()
                            .flex()
                            .gap_5()
                            .child(benchmark_param_field(
                                tr(&self.settings.language, "Warm-up tokens"),
                                tr(&self.settings.language, "Pre-run reduces cache cold-start effects"),
                                self.settings_text_control("benchmark-warmup", cx)
                                    .small()
                                    .text_size(px(13.0))
                                    .flex_1()
                                    .suffix(div().text_size(px(12.0)).text_color(rgb(MUTED)).child(tr(&self.settings.language, "tokens"))),
                            ))
                            .child(benchmark_param_field(
                                tr(&self.settings.language, "Runs per context"),
                                tr(&self.settings.language, "Repeated runs provide a more stable median"),
                                self.settings_text_control("benchmark-repeats", cx)
                                    .small()
                                    .text_size(px(13.0))
                                    .flex_1()
                                    .suffix(div().text_size(px(12.0)).text_color(rgb(MUTED)).child(tr(&self.settings.language, "times"))),
                            )),
                    )
                    .child(
                        div()
                            .flex()
                            .gap_5()
                            .child(benchmark_param_field(
                                tr(&self.settings.language, "Output tokens"),
                                tr(&self.settings.language, "Decode phase output token limit"),
                                self.settings_text_control("benchmark-max-tokens", cx)
                                    .small()
                                    .text_size(px(13.0))
                                    .flex_1()
                                    .suffix(div().text_size(px(12.0)).text_color(rgb(MUTED)).child(tr(&self.settings.language, "tokens"))),
                            ))
                            .child(benchmark_param_field(
                                tr(&self.settings.language, "Prefill step"),
                                tr(&self.settings.language, "Chunked-prefill window size"),
                                self.settings_text_control("benchmark-prefill-step", cx)
                                    .small()
                                    .text_size(px(13.0))
                                    .flex_1()
                                    .suffix(div().text_size(px(12.0)).text_color(rgb(MUTED)).child(tr(&self.settings.language, "step"))),
                            )),
                    )
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_2()
                            .child(
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_3()
                                    .child(div().text_size(px(13.0)).font_weight(FontWeight::MEDIUM).text_color(rgb(TEXT)).child(tr(&self.settings.language, "Contexts")))
                                    .child(div().text_size(px(12.0)).text_color(rgb(MUTED)).child(format!("{} {}", self.performance_contexts.len(), tr(&self.settings.language, "Selected contexts")))),
                            )
                            .child(div().flex().items_center().gap_2().text_size(px(13.0)).children(
                                [1024usize, 4096, 8192, 16384].map(|context| {
                                    let selected = self.performance_contexts.contains(&context);
                                    div()
                                        .px_2()
                                        .py_1()
                                        .rounded_md()
                                        .bg(rgb(if selected { 0xEFF6FF } else { WHITE }))
                                        .border_1()
                                        .border_color(rgb(if selected { 0xBFDBFE } else { BORDER }))
                                        .child(Checkbox::new(("benchmark-context", context))
                                            .small()
                                            .text_size(px(13.0))
                                            .label(format!("{}K", context / 1024))
                                            .checked(selected)
                                            .on_click(cx.listener(move |this, checked, _, cx| {
                                                if *checked {
                                                    if !this.performance_contexts.contains(&context) {
                                                        this.performance_contexts.push(context);
                                                        this.performance_contexts.sort_unstable();
                                                    }
                                                } else {
                                                    this.performance_contexts.retain(|value| *value != context);
                                                }
                                                cx.notify();
                                            })))
                                }),
                            )),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .justify_between()
                            .pt_3()
                            .border_t_1()
                            .border_color(rgb(BORDER))
                            .child(
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_1()
                                    .child(div().text_size(px(13.0)).font_weight(FontWeight::MEDIUM).text_color(rgb(TEXT)).child(tr(&self.settings.language, "Ignore EOS, fixed output length")))
                                    .child(div().text_size(px(12.0)).text_color(rgb(MUTED)).child(tr(&self.settings.language, "Force generation to the requested token count without stopping at EOS."))),
                            )
                            .child(
                                Switch::new("benchmark-ignore-eos")
                                    .checked(self.performance_ignore_eos)
                                    .on_click(cx.listener(|this, checked, _, cx| {
                                        this.performance_ignore_eos = *checked;
                                        cx.notify();
                                    })),
                            ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_4()
                            .child(div().text_size(px(16.0)).font_weight(FontWeight::SEMIBOLD).text_color(rgb(TEXT)).child(tr(&self.settings.language, "Latest report")))
                            .child(
                                TabBar::new("benchmark-report-tabs")
                                    .underline()
                                    .child(Tab::new(format!(
                                        "{}  {} {}",
                                        tr(&self.settings.language, "Results"),
                                        context_benchmark_rows(&self.report).len(),
                                        tr(&self.settings.language, "rows"),
                                    )))
                                    .child(Tab::new(tr(&self.settings.language, "Detailed report")))
                                    .selected_index(self.performance_report_tab)
                                    .on_click(cx.listener(|this, index, _, cx| {
                                        this.performance_report_tab = *index;
                                        cx.notify();
                                    })),
                            ),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                                    .when_some(benchmark_speculative_draft(&self.report), |row, (method, draft)| {
                                        row.child(
                                            div()
                                                .px_2()
                                                .py_1()
                                                .rounded_sm()
                                                .bg(rgb(if draft == "disabled" { 0xF1F5F9 } else { 0xECFDF5 }))
                                                .text_size(px(12.0))
                                                .text_color(rgb(if draft == "disabled" { SECONDARY } else { 0x059669 }))
                                                .child(if draft == "disabled" {
                                                    "Speculative decoding off".to_string()
                                                } else {
                                                    format!("{method} · {draft}")
                                                }),
                                        )
                                    })
                            .when(
                                self.performance_report_tab == 0
                                    && !context_benchmark_rows(&self.report).is_empty(),
                                |actions| actions.child(
                                app_button("export-benchmark-table")
                                    .label(tr(&self.settings.language, "Export CSV"))
                                    .text()
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        let Some(csv) = context_benchmark_csv(&this.report) else {
                                            return;
                                        };
                                        let directory = this
                                            .backend
                                            .lock()
                                            .ok()
                                            .map(|backend| backend.report_directory())
                                            .unwrap_or_else(|| std::env::current_dir().unwrap_or_default());
                                        let save_dialog = cx.prompt_for_new_path(
                                            &directory,
                                            Some("minimlx-benchmark.csv"),
                                        );
                                        let executor = cx.background_executor();
                                        let entity = cx.entity().downgrade();
                                        let export_task = executor.spawn(async move {
                                            match save_dialog.await {
                                                Ok(Ok(Some(path))) => {
                                                    let path_label = path.display().to_string();
                                                    fs::write(&path, csv)
                                                        .map(|_| Some(path_label))
                                                        .map_err(|error| error.to_string())
                                                }
                                                Ok(Ok(None)) => Ok(None),
                                                Ok(Err(error)) => Err(error.to_string()),
                                                Err(error) => Err(error.to_string()),
                                            }
                                        });
                                        cx.spawn(async move |_, cx| {
                                            let result = export_task.await;
                                            match result {
                                                Ok(Some(path)) => {
                                                    let _ = entity.update(cx, |this, cx| {
                                                        this.operation = format!("{}: {path}", tr(&this.settings.language, "Report exported"));
                                                        cx.notify();
                                                    });
                                                }
                                                Ok(None) => {}
                                                Err(error) => {
                                                    let _ = entity.update(cx, |this, cx| {
                                                        this.operation = format!("Could not export report: {error}");
                                                        cx.notify();
                                                    });
                                                }
                                            }
                                        })
                                        .detach();
                                    }))
                                ),
                            )
                            .child(
                                app_button("benchmark-history-archive")
                                    .label(format!(
                                        "{} ({})",
                                        tr(&self.settings.language, "History reports"),
                                        self.report_count
                                    ))
                                    .text()
                                    .on_click(cx.listener(|this, _, _, cx| {
                                        let path = this.backend.lock().ok().map(|backend| backend.report_directory());
                                        if let Some(path) = path {
                                            let _ = Command::new("open").arg(path).spawn();
                                        }
                                        cx.notify();
                                    })),
                            ),
                    ),
            )
            .child(
                div()
                    .flex()
                    .flex_col()
                    .min_w_0()
                    .flex_1()
                    .min_h(px(0.0))
                    .when(self.performance_report_tab == 0, |container| {
                        let rows = context_benchmark_rows(&self.report);
                        container.child(if self.report.is_empty() {
                            if self.benchmark_running {
                                div()
                                    .flex()
                                    .items_center()
                                    .gap_3()
                                    .min_h(px(108.0))
                                    .px_5()
                                    .bg(rgb(WHITE))
                                    .border_1()
                                    .border_color(rgb(BORDER))
                                    .rounded_md()
                                    .child(Indicator::new().small())
                                    .child(
                                        div()
                                            .flex()
                                            .flex_col()
                                            .gap_1()
                                            .child(
                                                div()
                                                    .text_size(px(14.0))
                                                    .font_weight(FontWeight::MEDIUM)
                                                    .text_color(rgb(TEXT))
                                                    .child(tr(&self.settings.language, "Testing…")),
                                            )
                                            .child(
                                                div()
                                                    .text_size(px(12.0))
                                                    .text_color(rgb(MUTED))
                                                    .child(tr(&self.settings.language, "Preparing the model and measuring selected context lengths.")),
                                            ),
                                    )
                                    .into_any_element()
                            } else if let Some(error) = self.benchmark_error.as_ref() {
                                div()
                                    .flex()
                                    .flex_col()
                                    .gap_1()
                                    .min_h(px(88.0))
                                    .justify_center()
                                    .px_5()
                                    .bg(rgb(0xFEF2F2))
                                    .border_1()
                                    .border_color(rgb(0xFECACA))
                                    .rounded_md()
                                    .child(
                                        div()
                                            .text_size(px(14.0))
                                            .font_weight(FontWeight::MEDIUM)
                                            .text_color(rgb(0xB91C1C))
                                            .child(tr(&self.settings.language, "Benchmark failed")),
                                    )
                                    .child(
                                        div()
                                            .text_size(px(12.0))
                                            .text_color(rgb(0x7F1D1D))
                                            .child(error.clone()),
                                    )
                                    .into_any_element()
                            } else {
                            div()
                                .py_4()
                                .text_color(rgb(MUTED))
                                .child(tr(
                                    &self.settings.language,
                                    "No report yet. Run the MiniMLX context matrix to create one.",
                                ))
                                .into_any_element()
                            }
                        } else if rows.is_empty() {
                            div()
                                .py_4()
                                .text_color(rgb(MUTED))
                                .child(tr(&self.settings.language, "No context results found in this report."))
                                .into_any_element()
                        } else {
                            div()
                                .flex()
                                .flex_col()
                                .min_w_0()
                                .bg(rgb(WHITE))
                                .border_1()
                                .border_color(rgb(BORDER))
                                .rounded_md()
                                .overflow_hidden()
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .h(px(42.0))
                                        .px_4()
                                        .bg(rgb(0xF8FAFC))
                                        .border_b_1()
                                        .border_color(rgb(BORDER))
                                        .text_size(px(12.0))
                                        .font_weight(FontWeight::SEMIBOLD)
                                        .text_color(rgb(MUTED))
                                        .children([
                                            "Context", "TTFT (ms)", "TPOT (ms)", "Prompt TPS (tok/s)",
                                            "Generation TPS (tok/s)", "E2E (s)", "Peak memory",
                                        ].map(|label| div().flex_1().min_w_0().child(tr(&self.settings.language, label)))),
                                )
                                .children(rows.into_iter().map(|row| {
                                    div()
                                        .flex()
                                        .items_center()
                                        .h(px(44.0))
                                        .px_4()
                                        .border_b_1()
                                        .border_color(rgb(BORDER))
                                        .text_size(px(13.0))
                                        .text_color(rgb(TEXT))
                                        .child(div().flex_1().min_w_0().text_color(rgb(SECONDARY)).child(format_context_label(&row.context)))
                                        .child(div().flex_1().min_w_0().child(row.ttft))
                                        .child(div().flex_1().min_w_0().child(row.tpot))
                                        .child(div().flex_1().min_w_0().child(row.prompt_tps))
                                        .child(div().flex_1().min_w_0().text_color(rgb(0x059669)).child(row.generation_tps))
                                        .child(div().flex_1().min_w_0().child(row.e2e_seconds))
                                        .child(div().flex_1().min_w_0().text_color(rgb(SECONDARY)).child(row.peak_memory))
                                }))
                                .into_any_element()
                        })
                    })
                    .when(self.performance_report_tab == 1, |container| {
                        let detailed_report = self.report.clone();
                        let copy_text = detailed_report.clone();
                        container.child(
                            div()
                                .flex()
                                .flex_col()
                                .min_w_0()
                                .flex_1()
                                .min_h(px(0.0))
                                .gap_2()
                                .pt_2()
                                .child(
                                    div()
                                        .flex()
                                        .justify_end()
                                        .child(
                                            app_button("copy-benchmark-log")
                                                .label(tr(&self.settings.language, "Copy report"))
                                                .on_click(cx.listener(move |_, _, _, cx| {
                                                    cx.write_to_clipboard(ClipboardItem::new_string(copy_text.clone()));
                                                })),
                                        ),
                                )
                                .child(
                                    div()
                                        .flex_1()
                                        .min_h(px(0.0))
                                        .min_w_0()
                                        .overflow_hidden()
                                        .border_1()
                                        .border_color(rgb(BORDER))
                                        .rounded_md()
                                        .child(if self.benchmark_running {
                                            div()
                                                .flex()
                                                .items_center()
                                                .gap_3()
                                                .p_4()
                                                .child(Indicator::new().small())
                                                .child(tr(&self.settings.language, "Testing…"))
                                                .into_any_element()
                                        } else if detailed_report.is_empty() {
                                            TextView::markdown(
                                                "benchmark-report-empty",
                                                tr(&self.settings.language, "No report yet. Run the MiniMLX context matrix to create one."),
                                                window,
                                                cx,
                                            )
                                            .selectable()
                                            .scrollable()
                                            .into_any_element()
                                        } else {
                                            TextView::markdown("benchmark-report-view", detailed_report, window, cx)
                                                .selectable()
                                                .scrollable()
                                                .into_any_element()
                                        }),
                                ),
                        )
                    }),
            )
    }

    fn run_benchmark_button(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let backend = self.backend.clone();
        let model_path = self.selected_model.clone();
        let model_options = model_inference_options_for_path(&self.settings, &model_path).cloned();
        let running = self.benchmark_running;
        app_button("run-minimlx-benchmark")
            .label(tr(
                &self.settings.language,
                if running {
                    "Testing…"
                } else {
                    "Run MiniMLX Test"
                },
            ))
            .primary()
            .loading(running)
            .disabled(running)
            .on_click(cx.listener(move |this, _, _, cx| {
                if this.benchmark_running {
                    return;
                }
                let parse_positive = |id: &str, label: &str| -> Result<usize, String> {
                    this.settings_text_inputs[id]
                        .read(cx)
                        .value()
                        .trim()
                        .parse::<usize>()
                        .ok()
                        .filter(|value| *value > 0)
                        .ok_or_else(|| format!("{label} 必须是大于 0 的整数"))
                };
                let benchmark_options = match (|| {
                    Ok::<_, String>(ContextBenchmarkOptions {
                        contexts: this.performance_contexts.clone(),
                        warmup: parse_positive("benchmark-warmup", "预热 Token 数")?,
                        repeats: parse_positive("benchmark-repeats", "重复次数")?,
                        max_tokens: parse_positive("benchmark-max-tokens", "生成 Token 数")?,
                        prefill_step: parse_positive("benchmark-prefill-step", "预填充步长")?,
                        ignore_eos: this.performance_ignore_eos,
                    })
                })() {
                    Ok(options) if !options.contexts.is_empty() => options,
                    Ok(_) => {
                        this.operation = tr(
                            &this.settings.language,
                            "Select at least one context length.",
                        );
                        cx.notify();
                        return;
                    }
                    Err(error) => {
                        this.operation = error;
                        cx.notify();
                        return;
                    }
                };
                this.benchmark_running = true;
                this.benchmark_error = None;
                this.report.clear();
                this.performance_report_tab = 0;
                this.operation = "Benchmark running · this may take several minutes…".into();
                this.api_running = false;
                this.loaded_model = None;
                cx.notify();
                let backend = backend.clone();
                let model_path = model_path.clone();
                let model_options = model_options.clone();
                let benchmark_options = benchmark_options.clone();
                let task = cx.background_executor().spawn(async move {
                    let Ok(mut backend) = backend.lock() else {
                        return (Err("runtime lock poisoned".to_string()), 0, None, false);
                    };
                    let report = backend.run_context_benchmark(
                        &model_path,
                        model_options.as_ref(),
                        &benchmark_options,
                    );
                    let report_count = backend.latest_report().0;
                    (
                        report,
                        report_count,
                        backend.loaded_model.clone(),
                        backend.api_running(),
                    )
                });
                let entity = cx.entity().downgrade();
                cx.spawn(async move |_, cx| {
                    let result = task.await;
                    let _ = entity.update(cx, |this, cx| {
                        let (result, report_count, loaded_model, api_running) = result;
                        this.report_count = report_count;
                        this.benchmark_running = false;
                        this.loaded_model = loaded_model;
                        this.api_running = api_running;
                        match result {
                            Ok(report) => {
                                this.report = report;
                                this.benchmark_error = None;
                                this.operation = "Benchmark complete · report saved".into();
                            }
                            Err(error) => {
                                this.report.clear();
                                this.benchmark_error = Some(error.clone());
                                this.operation = format!("Benchmark failed: {error}");
                            }
                        }
                        cx.notify();
                    });
                })
                .detach();
            }))
    }

    fn api_page(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .id("api-scroll")
            .flex()
            .flex_col()
            .flex_1()
            .overflow_y_scroll()
            .px_6()
            .pt_6()
            .pb_6()
            .gap_5()
            .child(self.page_heading(
                "API Server",
                "Configure and monitor the local inference endpoint.",
            ))
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .bg(rgb(SURFACE))
                    .border_1()
                    .border_color(rgb(BORDER))
                    .rounded_md()
                    .px_4()
                    .h(px(68.0))
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(
                                div()
                                    .text_size(px(13.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(rgb(TEXT))
                                    .child(tr(&self.settings.language, "Server status")),
                            )
                            .child(div().text_size(px(13.0)).text_color(rgb(MUTED)).child(
                                if self.api_running {
                                    format!("Listening · 127.0.0.1:{}", self.settings_port)
                                } else {
                                    "Stopped".to_string()
                                },
                            )),
                    )
                    .child(self.api_toggle_button(cx)),
            )
            .child(config_row(
                &self.settings.language,
                "Base URL",
                &format!("http://127.0.0.1:{}/v1", self.settings_port),
            ))
            .child(config_row(
                &self.settings.language,
                "Routes",
                "GET /v1/models · GET /metrics · POST /v1/chat/completions",
            ))
            .child(config_row(
                &self.settings.language,
                "Compatibility",
                "OpenAI Chat Completions · text and local/data-URI images",
            ))
            .child(config_row(
                &self.settings.language,
                "Network access",
                "Loopback only · unauthenticated local API",
            ))
            .child(config_row(
                &self.settings.language,
                "Text streaming",
                "SSE-compatible · buffered until generation completes",
            ))
    }

    fn open_api_model_picker(&mut self, window: &mut Window, cx: &mut Context<Self>) {
        let model_options = self
            .models
            .iter()
            .filter(|model| !is_speculative_draft_entry(model))
            .map(|model| BenchmarkModelOption {
                title: format!("{}  ·  {}", model.name, format_bytes(model.size_bytes)),
                path: model.path.clone(),
            })
            .collect::<Vec<_>>();
        if model_options.is_empty() {
            let language = self.settings.language.clone();
            let entity = cx.entity().downgrade();
            window.open_modal(cx, move |modal, _, _| {
                let entity = entity.clone();
                modal
                    .title(tr(&language, "No runnable MiniMLX models are available"))
                    .width(px(480.0))
                    .show_close(false)
                    .child(
                        div()
                            .flex()
                            .flex_col()
                            .gap_4()
                            .child(
                                div()
                                    .text_size(px(13.0))
                                    .text_color(rgb(SECONDARY))
                                    .child(tr(
                                        &language,
                                        "Download or add a model in Models before starting the service.",
                                    )),
                            )
                            .child(
                                div()
                                    .flex()
                                    .justify_end()
                                    .gap_2()
                                    .child(
                                        app_button("api-model-picker-cancel")
                                            .label(tr(&language, "Cancel"))
                                            .on_click(|_, window, cx| window.close_modal(cx)),
                                    )
                                    .child(
                                        app_button("api-model-picker-go-models")
                                            .label(tr(&language, "Go to Models"))
                                            .primary()
                                            .on_click(move |_, window, cx| {
                                                let _ = entity.update(cx, |this, cx| {
                                                    window.close_modal(cx);
                                                    this.page = Page::Models;
                                                    cx.notify();
                                                });
                                            }),
                                    ),
                            ),
                    )
            });
            return;
        }

        let language = self.settings.language.clone();
        let model_dropdown = cx.new(|cx| {
            DropdownState::new(model_options, None, window, cx)
        });
        let entity = cx.entity().downgrade();
        window.open_modal(cx, move |modal, _, _| {
            let dropdown = model_dropdown.clone();
            let start_dropdown = model_dropdown.clone();
            let start_entity = entity.clone();
            modal
                .title(tr(&language, "Choose a model to start the service"))
                .width(px(560.0))
                .show_close(false)
                .child(
                    div()
                        .flex()
                        .flex_col()
                        .gap_3()
                        .child(
                            div()
                                .text_size(px(13.0))
                                .text_color(rgb(SECONDARY))
                                .child(tr(
                                    &language,
                                    "Select one of your MiniMLX models. The service will start after you choose.",
                                )),
                        )
                        .child(
                            Dropdown::new(&dropdown)
                                .small()
                                .placeholder(tr(&language, "Select model"))
                                .w_full(),
                        )
                        .child(
                            div()
                                .flex()
                                .justify_end()
                                .gap_2()
                                .child(
                                    app_button("api-model-picker-cancel")
                                        .label(tr(&language, "Cancel"))
                                        .on_click(|_, window, cx| window.close_modal(cx)),
                                )
                                .child(
                                    app_button("api-model-picker-start")
                                        .label(tr(&language, "Start Service"))
                                        .primary()
                                        .on_click(move |_, window, cx| {
                                            let Some(path) = start_dropdown
                                                .read(cx)
                                                .selected_value()
                                                .cloned()
                                            else {
                                                return;
                                            };
                                            let _ = start_entity.update(cx, |this, cx| {
                                                window.close_modal(cx);
                                                this.begin_api_start(path, cx);
                                            });
                                        }),
                                ),
                        ),
                )
        });
    }

    fn begin_api_start(
        &mut self,
        model_path: String,
        cx: &mut Context<Self>,
    ) {
        let Some(entry) = self
            .models
            .iter()
            .find(|model| same_model_path(&model.path, &model_path))
        else {
            self.operation = tr(
                &self.settings.language,
                "Only models in the MiniMLX model list can be started",
            );
            cx.notify();
            return;
        };
        let model_options = model_inference_options_for_path(&self.settings, &entry.path).cloned();
        let backend = self.backend.clone();
        self.operation = "Starting API server…".into();
        cx.notify();
        let task = cx.background_executor().spawn(async move {
            let Ok(mut backend) = backend.lock() else {
                return Err("runtime lock poisoned".to_string());
            };
            backend.start_api_with_options(&model_path, model_options.as_ref())?;
            Ok((
                backend.models(),
                backend.loaded_model.clone(),
                backend.api_running(),
            ))
        });
        let entity = cx.entity().downgrade();
        cx.spawn(async move |_, cx| {
            let result = task.await;
            let _ = entity.update(cx, |this, cx| {
                match result {
                    Ok((models, loaded_model, api_running)) => {
                        this.operation = "API server started".into();
                        this.models = models;
                        this.loaded_model = loaded_model.clone();
                        this.api_running = api_running;
                        if let Some(path) = loaded_model {
                            this.selected_model = path.clone();
                            this.selected_library_model = Some(path);
                        }
                    }
                    Err(error) => {
                        this.operation = format!("API operation failed: {error}");
                        this.api_running = this
                            .backend
                            .lock()
                            .ok()
                            .is_some_and(|mut backend| backend.api_running());
                    }
                }
                cx.notify();
            });
        })
        .detach();
    }

    fn api_toggle_button(&self, cx: &mut Context<Self>) -> impl IntoElement {
        let backend = self.backend.clone();
        let model_path = self
            .loaded_model
            .as_ref()
            .filter(|path| {
                self.models
                    .iter()
                    .any(|model| {
                        same_model_path(&model.path, path) && !is_speculative_draft_entry(model)
                    })
            })
            .cloned()
            .unwrap_or_default();
        let model_options = model_inference_options_for_path(&self.settings, &model_path).cloned();
        let starting = !self.api_running;
        app_button("toggle-api-server")
            .label(if starting {
                "Start Service"
            } else {
                "Stop Service"
            })
            .when(starting, |button| button.primary())
            .on_click(cx.listener(move |this, _, window, cx| {
                if this.operation.ends_with('…') {
                    return;
                }
                if starting && model_path.is_empty() {
                    this.open_api_model_picker(window, cx);
                    return;
                }
                this.operation = if starting {
                    "Starting API server…"
                } else {
                    "Stopping API server…"
                }
                .into();
                cx.notify();
                let backend = backend.clone();
                let model_path = model_path.clone();
                let model_options = model_options.clone();
                let task = cx.background_executor().spawn(async move {
                    let Ok(mut backend) = backend.lock() else {
                        return Err("runtime lock poisoned".to_string());
                    };
                    if starting {
                        backend
                            .start_api_with_options(&model_path, model_options.as_ref())
                            .map(|_| "API server started".to_string())?;
                    } else {
                        backend.stop_api();
                    }
                    let message = if starting {
                        "API server started"
                    } else {
                        "API server stopped"
                    };
                    Ok((
                        message.to_string(),
                        backend.models(),
                        backend.loaded_model.clone(),
                        backend.api_running(),
                    ))
                });
                let entity = cx.entity().downgrade();
                cx.spawn(async move |_, cx| {
                    let result = task.await;
                    let _ = entity.update(cx, |this, cx| {
                        match result {
                            Ok((message, models, loaded_model, api_running)) => {
                                this.operation = message;
                                this.models = models;
                                this.loaded_model = loaded_model.clone();
                                this.api_running = api_running;
                                if let Some(path) = loaded_model {
                                    this.selected_model = path.clone();
                                    this.selected_library_model = Some(path);
                                }
                            }
                            Err(error) => {
                                this.operation = format!("API operation failed: {error}");
                                this.api_running = this
                                    .backend
                                    .lock()
                                    .ok()
                                    .is_some_and(|mut backend| backend.api_running());
                            }
                        }
                        cx.notify();
                    });
                })
                .detach();
            }))
    }

    fn settings_page(&mut self, cx: &mut Context<Self>) -> impl IntoElement {
        let filtered: Vec<_> = SettingsCategory::ALL
            .into_iter()
            .filter(|category| {
                self.settings_search.is_empty()
                    || category
                        .title(&self.settings.language)
                        .to_lowercase()
                        .contains(&self.settings_search.to_lowercase())
                    || category
                        .title("en")
                        .to_lowercase()
                        .contains(&self.settings_search.to_lowercase())
            })
            .collect();
        let port = self.settings_port.clone();
        let max_tokens = self.settings_max_tokens.clone();
        let context_window = self.settings_context_window.clone();
        let temperature = self.settings_temperature.clone();
        let top_p = self.settings_top_p.clone();
        let top_k = self.settings_top_k.clone();
        let repetition_penalty = self.settings_repetition_penalty.clone();
        let data_path = self.settings_data_path.clone();
        let huggingface_endpoint = self.settings_huggingface_endpoint.clone();
        let http_proxy = self.settings_http_proxy.clone();
        let https_proxy = self.settings_https_proxy.clone();
        let no_proxy = self.settings_no_proxy.clone();
        let ca_bundle = self.settings_ca_bundle.clone();
        let running = self.api_running;
        let model_path = self.loaded_model.clone().unwrap_or_default();
        let backend = self.backend.clone();
        div()
            .id("settings-root")
            .flex()
            .flex_col()
            .flex_1()
            .min_h(px(0.0))
            .px_6()
            .pt_6()
            .pb_6()
            .gap_4()
            .child(self.page_heading(
                "Settings",
                "Configure the local inference service and runtime preferences.",
            ))
            .child(
                div()
                    .flex()
                    .flex_1()
                    .min_h(px(0.0))
                    .gap_5()
                    .child(
                        div()
                            .w(px(205.0))
                            .flex_shrink_0()
                            .flex()
                            .flex_col()
                            .gap_1()
                            .child(self.settings_search_input(cx))
                            .child(
                                div()
                                    .mt_3()
                                    .mb_1()
                                    .px_2()
                                    .text_size(px(13.0))
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(rgb(MUTED))
                                    .child("MINIMLX"),
                            )
                            .children(filtered.into_iter().map(|category| {
                                self.settings_category_button(category, cx)
                            })),
                    )
                    .child(
                        div()
                            .id("settings-category-scroll")
                            .flex()
                            .flex_col()
                            .flex_1()
                            .min_w(px(0.0))
                            .min_h(px(0.0))
                            .scrollbar_width(px(8.0))
                            .gap_4()
                            .child(self.settings_category_content(cx)),
                    ),
            )
            .child(
                div()
                    .flex()
                    .items_center()
                    .justify_between()
                    .pt_3()
                    .border_t_1()
                    .border_color(rgb(BORDER))
                    .child(
                        div()
                            .text_size(px(14.0))
                            .text_color(rgb(if self.settings_message.is_empty() { MUTED } else { SECONDARY }))
                            .child(if self.settings_message.is_empty() {
                                if self.settings_dirty { tr(&self.settings.language, "Unsaved changes") } else { tr(&self.settings.language, "Settings are saved in your user profile") }
                            } else {
                                tr(&self.settings.language, &self.settings_message)
                            }),
                    )
                    .child(
                        div()
                            .flex()
                            .items_center()
                            .gap_2()
                            .child(
                                app_button("reset-settings")
                                    .label(tr(&self.settings.language, "Reset"))
                                    .on_click(cx.listener(|this, _, window, cx| {
                                        let defaults = DesktopSettings::default();
                                        this.settings_port = defaults.port.to_string();
                                        this.settings_max_tokens = defaults.max_tokens.to_string();
                                        this.settings_context_window = defaults.max_context_tokens.to_string();
                                        this.settings_temperature = defaults.temperature.to_string();
                                        this.settings_top_p = defaults.top_p.to_string();
                                        this.settings_top_k = defaults.top_k.to_string();
                                        this.settings_repetition_penalty = defaults.repetition_penalty.to_string();
                                        this.settings_data_path = defaults.data_path.clone();
                                        this.settings_huggingface_endpoint = defaults.huggingface_endpoint.clone();
                                        this.settings_http_proxy = defaults.http_proxy.clone();
                                        this.settings_https_proxy = defaults.https_proxy.clone();
                                        this.settings_no_proxy = defaults.no_proxy.clone();
                                        this.settings_ca_bundle = defaults.ca_bundle.clone();
                                        this.settings = defaults;
                                        for (id, value) in this.settings_input_values() {
                                            this.sync_settings_text_input(&id, &value, window, cx);
                                        }
                                        this.settings_dirty = true;
                                        this.settings_message.clear();
                                        cx.notify();
                                    })),
                            )
                            .child(
                                app_button("apply-settings")
                                    .label(tr(&self.settings.language, "Apply"))
                                    .when(self.settings_dirty, |button| button.primary())
                                    .disabled(!self.settings_dirty)
                                    .on_click(cx.listener(move |this, _, _, cx| {
                                        let Ok(port) = port.parse::<u16>() else {
                                            this.settings_message = "Port must be a number between 1024 and 65535".into();
                                            cx.notify();
                                            return;
                                        };
                                        let Ok(max_tokens) = max_tokens.parse::<u32>() else {
                                            this.settings_message = "Maximum tokens must be a positive whole number".into();
                                            cx.notify();
                                            return;
                                        };
                                        let Ok(context_window) = context_window.parse::<u32>() else {
                                            this.settings_message = "Context window must be a positive whole number".into();
                                            cx.notify();
                                            return;
                                        };
                                        let Ok(temperature) = temperature.parse::<f32>() else {
                                            this.settings_message = "Temperature must be a finite, non-negative number".into();
                                            cx.notify();
                                            return;
                                        };
                                        let Ok(top_p) = top_p.parse::<f32>() else {
                                            this.settings_message = "Top P must be a number from 0 to 1".into();
                                            cx.notify();
                                            return;
                                        };
                                        let Ok(top_k) = top_k.parse::<u32>() else {
                                            this.settings_message = "Top K must be a non-negative whole number".into();
                                            cx.notify();
                                            return;
                                        };
                                        let Ok(repetition_penalty) = repetition_penalty.parse::<f32>() else {
                                            this.settings_message = "Repetition penalty must be greater than 0".into();
                                            cx.notify();
                                            return;
                                        };
                                        let endpoint = huggingface_endpoint.trim().trim_end_matches('/');
                                        if !(endpoint.starts_with("https://") || endpoint.starts_with("http://"))
                                            || endpoint.chars().any(char::is_whitespace)
                                            || endpoint.len() > 512
                                        {
                                            this.settings_message = "Enter a valid Hugging Face endpoint URL".into();
                                            cx.notify();
                                            return;
                                        }
                                        let data_path = data_path.trim();
                                        let data_path = if let Some(stripped) = data_path.strip_prefix("~/") {
                                            std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default().join(stripped)
                                        } else {
                                            PathBuf::from(data_path)
                                        };
                                        if !(1024..=65535).contains(&port)
                                            || !(1..=1_000_000).contains(&max_tokens)
                                            || !(1..=1_000_000).contains(&context_window)
                                            || !valid_temperature(temperature)
                                            || !top_p.is_finite() || !(0.0..=1.0).contains(&top_p)
                                            || !repetition_penalty.is_finite() || repetition_penalty <= 0.0
                                        {
                                            this.settings_message = "One or more values are outside the supported range".into();
                                            cx.notify();
                                            return;
                                        }
                                        let mut settings = this.settings.clone();
                                        settings.port = port;
                                        settings.max_tokens = max_tokens;
                                        settings.max_context_tokens = context_window;
                                        settings.temperature = temperature;
                                        settings.top_p = top_p;
                                        settings.top_k = top_k;
                                        settings.repetition_penalty = repetition_penalty;
                                        settings.data_path = data_path.to_string_lossy().to_string();
                                        settings.huggingface_endpoint = endpoint.to_string();
                                        settings.http_proxy = http_proxy.trim().to_string();
                                        settings.https_proxy = https_proxy.trim().to_string();
                                        settings.no_proxy = no_proxy.trim().to_string();
                                        settings.ca_bundle = ca_bundle.trim().to_string();
                                        if settings.no_proxy.chars().any(char::is_control) {
                                            this.settings_message = "No proxy list must not contain control characters".into();
                                            cx.notify();
                                            return;
                                        }
                                        if !settings.ca_bundle.is_empty() {
                                            let ca_path = expand_user_directory(&settings.ca_bundle);
                                            if !ca_path.is_file() {
                                                this.settings_message = format!("CA bundle file does not exist: {}", ca_path.display());
                                                cx.notify();
                                                return;
                                            }
                                            settings.ca_bundle = ca_path.display().to_string();
                                        }
                                        for (label, value) in [("HTTP", &settings.http_proxy), ("HTTPS", &settings.https_proxy)] {
                                            if !value.is_empty() {
                                                let Some((scheme, authority)) = value.split_once("://") else {
                                                    this.settings_message = format!("{label} proxy must use http:// or https://");
                                                    cx.notify();
                                                    return;
                                                };
                                                if !matches!(scheme, "http" | "https") || authority.is_empty() || authority.contains('@') || value.chars().any(char::is_whitespace) {
                                                    this.settings_message = format!("Enter a valid {label} proxy URL without embedded credentials");
                                                    cx.notify();
                                                    return;
                                                }
                                            }
                                        }
                                        if let Err(error) = fs::create_dir_all(&data_path) {
                                            this.settings_message = format!("Could not create data directory: {error}");
                                            cx.notify();
                                            return;
                                        }
                                        let api_key = if settings.auth_enabled { keychain_api_key() } else { None };
                                        if settings.auth_enabled && api_key.is_none() {
                                            this.settings_message = "API-key authentication requires a key. Generate one in Security first.".into();
                                            cx.notify();
                                            return;
                                        }
                                        if settings.bind_lan && api_key.is_none() {
                                            this.settings_message = "LAN access requires an API key. Generate one in Security first.".into();
                                            cx.notify();
                                            return;
                                        }
                                        if settings.model_directories.is_empty() {
                                            this.settings_message = "Keep at least one model directory configured".into();
                                            cx.notify();
                                            return;
                                        }
                                        for (index, configured_directory) in settings.model_directories.iter_mut().enumerate() {
                                            let directory = expand_user_directory(configured_directory);
                                            if index == 0 {
                                                if let Err(error) = fs::create_dir_all(&directory) {
                                                    this.settings_message = format!("Could not create model directory {}: {error}", directory.display());
                                                    cx.notify();
                                                    return;
                                                }
                                                *configured_directory = directory.display().to_string();
                                            } else {
                                                match validate_external_model_directory(&directory) {
                                                    Ok(canonical) => *configured_directory = canonical.display().to_string(),
                                                    Err(error) => {
                                                        this.settings_message = format!("Could not add external model folder: {error}");
                                                        cx.notify();
                                                        return;
                                                    }
                                                }
                                            }
                                        }
                                        if let Err(error) = persist_desktop_settings(&settings) {
                                            this.settings_message = format!("Could not save settings: {error}");
                                            cx.notify();
                                            return;
                                        }
                                        this.settings = settings.clone();
                                        this.settings_dirty = false;
                                        this.settings_message = if running {
                                            "Applying settings and restarting the service…".into()
                                        } else {
                                            "Settings saved · used the next time the service starts".into()
                                        };
                                        cx.notify();
                                        let backend = backend.clone();
                                        let model_path = model_path.clone();
                                        let model_options =
                                            model_inference_options_for_path(&settings, &model_path)
                                                .cloned();
                                        let task = cx.background_executor().spawn(async move {
                                            let Ok(mut backend) = backend.lock() else { return Err("runtime lock poisoned".to_string()); };
                                            backend.configure_storage(
                                                settings.model_directories.iter().map(PathBuf::from).collect(),
                                            )?;
                                            backend.configure_download(
                                                settings.huggingface_endpoint.clone(),
                                                settings.http_proxy.clone(),
                                                settings.https_proxy.clone(),
                                                settings.no_proxy.clone(),
                                                settings.ca_bundle.clone(),
                                            )?;
                                            backend.configure_api(
                                                if settings.bind_lan { "0.0.0.0" } else { "127.0.0.1" }.into(),
                                                settings.port,
                                                settings.max_tokens,
                                                settings.max_context_tokens,
                                                settings.temperature,
                                                settings.top_p,
                                                settings.top_k,
                                                settings.repetition_penalty,
                                                settings.log_level.clone(),
                                                settings.record_usage_history,
                                                data_path.join("usage.sqlite3"),
                                                settings.auth_enabled,
                                                api_key,
                                            )?;
                                            if running && !model_path.is_empty() {
                                                backend.stop_api();
                                                backend.start_api_with_options(&model_path, model_options.as_ref())?;
                                            }
                                            Ok::<(), String>(())
                                        });
                                        let entity = cx.entity().downgrade();
                                        cx.spawn(async move |_, cx| {
                                            let result = task.await;
                                            let _ = entity.update(cx, |this, cx| {
                                                this.settings_message = result.map(|_| "Settings applied".to_string()).unwrap_or_else(|error| format!("Settings saved, but service restart failed: {error}"));
                                                this.api_running = this.backend.lock().ok().is_some_and(|mut backend| backend.api_running());
                                                this.models = this.backend.lock().map(|backend| backend.models()).unwrap_or_default();
                                                cx.notify();
                                            });
                                        }).detach();
                                    })),
                            ),
                    ),
            )
    }

    fn settings_search_input(&self, _cx: &mut Context<Self>) -> impl IntoElement {
        let input = self.settings_text_inputs["settings-search"].clone();
        div()
            .flex()
            .items_center()
            .gap_2()
            .px_2()
            .h(px(32.0))
            .rounded_sm()
            .border_1()
            .border_color(rgb(BORDER))
            .bg(rgb(SURFACE))
            .child(
                TextInput::new(&input)
                    .w_full()
                    .appearance(false)
                    .prefix(div().text_size(px(14.0)).text_color(rgb(MUTED)).child("⌕")),
            )
    }

    fn settings_category_button(
        &self,
        category: SettingsCategory,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let selected = self.settings_category == category;
        app_button(SharedString::from(format!(
            "settings-category-{}",
            category.title(&self.settings.language)
        )))
        .label(category.title(&self.settings.language))
        .selected(selected)
        .ghost()
        .on_click(cx.listener(move |this, _, _, cx| {
            this.settings_category = category;
            cx.notify();
        }))
    }

    fn settings_number_row(
        &self,
        label: &'static str,
        _value: &str,
        _setting: NumericSetting,
        detail: &'static str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let field_id = format!("setting-value-{label}");
        div()
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .min_h(px(62.0))
            .py_2()
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_size(px(13.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, label)),
                    )
                    .child(
                        div()
                            .text_size(px(13.0))
                            .text_color(rgb(MUTED))
                            .child(tr(&self.settings.language, detail)),
                    ),
            )
            .child(self.settings_text_control(&field_id, cx).w(px(280.0)))
    }

    fn settings_endpoint_row(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .min_h(px(62.0))
            .py_2()
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(div().text_size(px(13.0)).font_weight(FontWeight::MEDIUM).text_color(rgb(TEXT)).child(tr(&self.settings.language, "Hugging Face endpoint")))
                    .child(div().text_size(px(13.0)).text_color(rgb(MUTED)).child(tr(&self.settings.language, "Used for model metadata and file resolution. Xet/CAS is disabled; the Hub may still redirect file bytes to its storage host."))),
            )
            .child(self.settings_text_control("setting-huggingface-endpoint", cx).w(px(340.0)))
    }

    fn settings_download_text_row(
        &self,
        label: &'static str,
        detail: &'static str,
        _setting: DownloadTextSetting,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let field_id = format!("setting-download-{label}");
        div()
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .min_h(px(62.0))
            .py_2()
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_size(px(13.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, label)),
                    )
                    .child(
                        div()
                            .text_size(px(12.0))
                            .text_color(rgb(MUTED))
                            .child(tr(&self.settings.language, detail)),
                    ),
            )
            .child(self.settings_text_control(&field_id, cx).w(px(340.0)))
    }

    fn settings_boolean_row(
        &self,
        label: &'static str,
        detail: &'static str,
        setting: BooleanSetting,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let enabled = match setting {
            BooleanSetting::BindLan => self.settings.bind_lan,
            BooleanSetting::Authentication => self.settings.auth_enabled,
            BooleanSetting::StartService => self.settings.start_service_on_launch,
            BooleanSetting::UsageHistory => self.settings.record_usage_history,
        };
        div()
            .flex()
            .items_center()
            .justify_between()
            .gap_4()
            .min_h(px(58.0))
            .py_2()
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_size(px(13.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, label)),
                    )
                    .child(
                        div()
                            .text_size(px(13.0))
                            .text_color(rgb(MUTED))
                            .child(tr(&self.settings.language, detail)),
                    ),
            )
            .child(
                Switch::new(SharedString::from(format!("toggle-{label}")))
                    .checked(enabled)
                    .on_click(cx.listener(move |this, checked, _, cx| {
                        match setting {
                            BooleanSetting::BindLan => this.settings.bind_lan = *checked,
                            BooleanSetting::Authentication => this.settings.auth_enabled = *checked,
                            BooleanSetting::StartService => {
                                this.settings.start_service_on_launch = *checked
                            }
                            BooleanSetting::UsageHistory => {
                                this.settings.record_usage_history = *checked
                            }
                        }
                        this.settings_dirty = true;
                        this.settings_message.clear();
                        cx.notify();
                    })),
            )
    }

    fn settings_api_key_row(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .justify_between()
            .min_h(px(58.0))
            .py_2()
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .gap_1()
                    .child(
                        div()
                            .text_size(px(13.0))
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(rgb(TEXT))
                            .child(tr(&self.settings.language, "API key")),
                    )
                    .child(div().text_size(px(13.0)).text_color(rgb(MUTED)).child(
                        if self.api_key_ready {
                            tr(&self.settings.language, "Stored in macOS Keychain")
                        } else {
                            tr(&self.settings.language, "No key generated")
                        },
                    )),
            )
            .child(
                app_button("generate-api-key")
                    .label(if self.api_key_ready {
                        tr(&self.settings.language, "Regenerate")
                    } else {
                        tr(&self.settings.language, "Generate key")
                    })
                    .outline()
                    .on_click(cx.listener(|this, _, _, cx| {
                        let output = Command::new("/usr/bin/openssl")
                            .args(["rand", "-hex", "32"])
                            .output();
                        let result = output
                            .map_err(|error| error.to_string())
                            .and_then(|output| {
                                if !output.status.success() {
                                    return Err(String::from_utf8_lossy(&output.stderr)
                                        .trim()
                                        .to_string());
                                }
                                let key =
                                    String::from_utf8_lossy(&output.stdout).trim().to_string();
                                store_keychain_api_key(&key)?;
                                Ok(key)
                            });
                        match result {
                            Ok(key) => {
                                cx.write_to_clipboard(ClipboardItem::new_string(key));
                                this.api_key_ready = true;
                                this.settings.auth_enabled = true;
                                this.settings_dirty = true;
                                this.settings_message = tr(
                                    &this.settings.language,
                                    "New API key saved to Keychain and copied to clipboard",
                                );
                            }
                            Err(error) => {
                                this.settings_message = format!(
                                    "{}: {error}",
                                    tr(&this.settings.language, "Could not create API key")
                                )
                            }
                        }
                        cx.notify();
                    })),
            )
    }

    fn settings_log_level_row(&self) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .justify_between()
            .min_h(px(58.0))
            .py_2()
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(rgb(TEXT))
                    .child(tr(&self.settings.language, "Log level")),
            )
            .child(Dropdown::new(&self.log_level_dropdown).w(px(180.0)))
    }

    fn settings_directory_input(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .gap_2()
            .child(
                self.settings_text_control("model-directory-input", cx)
                    .flex_1(),
            )
            .child(
                app_button("add-model-directory")
                    .label(tr(&self.settings.language, "Add external folder"))
                    .on_click(cx.listener(|this, _, window, cx| {
                        let value = this.settings_new_model_directory.trim().to_string();
                        if value.is_empty() {
                            this.settings_message = tr(
                                &this.settings.language,
                                "Enter a model directory path first",
                            );
                        } else {
                            let path = expand_user_directory(&value);
                            match validate_external_model_directory(&path) {
                                Ok(canonical) => {
                                    let canonical_string = canonical.display().to_string();
                                    let already_added = this.settings.model_directories.iter().any(|entry| {
                                        let existing = expand_user_directory(entry);
                                        existing.canonicalize().unwrap_or(existing) == canonical
                                    });
                                    if already_added {
                                        this.settings_message = tr(
                                            &this.settings.language,
                                            "This external model folder is already added.",
                                        );
                                    } else {
                                        this.settings.model_directories.push(canonical_string);
                                        this.settings_dirty = true;
                                        this.settings_message = tr(
                                            &this.settings.language,
                                            "External folders are referenced in place; files are not copied or moved.",
                                        );
                                    }
                                }
                                Err(error) => {
                                    this.settings_message = if error.contains(".cache") {
                                        tr(&this.settings.language, "Model folders inside .cache cannot be added.")
                                    } else {
                                        tr(&this.settings.language, "No supported model found in that folder. Add a folder containing a MiniMLX-compatible model.")
                                    };
                                }
                            }
                        this.settings_new_model_directory.clear();
                        if let Some(input) = this.settings_text_inputs.get("model-directory-input") {
                            input.update(cx, |input, cx| input.set_value("", window, cx));
                        }
                        }
                        cx.notify();
                    })),
            )
    }

    fn settings_directory_suggestions(&self, cx: &mut Context<Self>) -> impl IntoElement {
        div()
            .flex()
            .items_center()
            .gap_3()
            .pt_1()
            .text_size(px(12.0))
            .text_color(rgb(MUTED))
            .child(tr(&self.settings.language, "Common model folders"))
            .child(
                app_button("suggest-omlx-models-folder")
                    .label("oMLX  ~/.omlx/models")
                    .text()
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.sync_settings_text_input(
                            "model-directory-input",
                            "~/.omlx/models",
                            window,
                            cx,
                        );
                        this.settings_message.clear();
                        cx.notify();
                    })),
            )
            .child(
                app_button("suggest-lmstudio-models-folder")
                    .label("LM Studio  ~/.lmstudio/models")
                    .text()
                    .on_click(cx.listener(|this, _, window, cx| {
                        this.sync_settings_text_input(
                            "model-directory-input",
                            "~/.lmstudio/models",
                            window,
                            cx,
                        );
                        this.settings_message.clear();
                        cx.notify();
                    })),
            )
    }

    fn settings_path_row(
        &self,
        label: &'static str,
        detail: &'static str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        div()
            .flex()
            .flex_col()
            .gap_1()
            .py_2()
            .border_b_1()
            .border_color(rgb(BORDER))
            .child(
                div()
                    .text_size(px(13.0))
                    .font_weight(FontWeight::MEDIUM)
                    .text_color(rgb(TEXT))
                    .child(tr(&self.settings.language, label)),
            )
            .child(
                div()
                    .text_size(px(13.0))
                    .text_color(rgb(MUTED))
                    .child(tr(&self.settings.language, detail)),
            )
            .child(
                self.settings_text_control("settings-data-path", cx)
                    .w_full(),
            )
    }

    fn settings_language_button(
        &self,
        language: &'static str,
        label: &'static str,
        cx: &mut Context<Self>,
    ) -> impl IntoElement {
        let selected = self.settings.language == language;
        app_button(SharedString::from(format!("language-{language}")))
            .label(tr(&self.settings.language, label))
            .selected(selected)
            .ghost()
            .on_click(cx.listener(move |this, _, _, cx| {
                let previous = this.settings.language.clone();
                this.settings.language = language.to_string();
                match persist_language_preference(language) {
                    Ok(()) => this.settings_message = tr(language, "Language saved"),
                    Err(error) => {
                        this.settings.language = previous;
                        this.settings_message = format!(
                            "{}: {error}",
                            tr(
                                &this.settings.language,
                                "Could not save language preference"
                            )
                        );
                    }
                }
                cx.notify();
            }))
    }

    fn settings_category_content(&self, cx: &mut Context<Self>) -> AnyElement {
        let mut content = div().flex().flex_col().gap_4().child(
            div().pb_3().border_b_1().border_color(rgb(BORDER)).child(
                div()
                    .text_size(px(17.0))
                    .font_weight(FontWeight::SEMIBOLD)
                    .text_color(rgb(TEXT))
                    .child(self.settings_category.title(&self.settings.language)),
            ),
        );
        match self.settings_category {
            SettingsCategory::General => {
                content = content.child(settings_panel(vec![
                    settings_text_row(
                        &tr(&self.settings.language, "Language"),
                        &tr(
                            &self.settings.language,
                            if self.settings.language == "zh-CN" {
                                "简体中文"
                            } else {
                                "English"
                            },
                        ),
                        &tr(
                            &self.settings.language,
                            "Choose the language used by the MiniMLX interface.",
                        ),
                    ),
                    div()
                        .flex()
                        .gap_2()
                        .py_3()
                        .child(self.settings_language_button("en", "English", cx))
                        .child(self.settings_language_button("zh-CN", "简体中文", cx))
                        .into_any_element(),
                ]));
            }
            SettingsCategory::Server => {
                let host = if self.settings.bind_lan {
                    "0.0.0.0"
                } else {
                    "127.0.0.1"
                };
                content = content
                    .child(settings_panel(vec![
                        self.settings_boolean_row("Listen on LAN", "Expose the service on all network interfaces; API-key authentication is required", BooleanSetting::BindLan, cx).into_any_element(),
                        self.settings_number_row("Port", &self.settings_port, NumericSetting::Port, "Default 18765 · service restarts when applied", cx).into_any_element(),
                        settings_text_row(&tr(&self.settings.language, "OpenAI-compatible API"), &format!("http://{host}:{}/v1", self.settings_port), &tr(&self.settings.language, "Chat Completions and model listing")),
                        settings_text_row(&tr(&self.settings.language, "Anthropic Messages"), &format!("http://{host}:{}/v1/messages", self.settings_port), &tr(&self.settings.language, "POST /v1/messages · text and image requests; tool use is not advertised")),
                        settings_text_row(&tr(&self.settings.language, "Health check"), &format!("http://{host}:{}/health", self.settings_port), &tr(&self.settings.language, "GET /health")),
                        settings_text_row(&tr(&self.settings.language, "Prometheus metrics"), &format!("http://{host}:{}/metrics", self.settings_port), &tr(&self.settings.language, "GET /metrics · Prometheus text exposition")),
                    ]))
                    .child(settings_note(&tr(&self.settings.language, "Changes apply to the next server start. If the service is running, Apply restarts it with the new settings.")));
            }
            SettingsCategory::Network => {
                content = content
                    .child(settings_panel(vec![
                        self.settings_endpoint_row(cx).into_any_element(),
                        settings_group_heading(&self.settings.language, "Proxy", "Outbound HTTP routing. Blank proxy fields mean direct connection; system proxy variables are ignored."),
                        self.settings_download_text_row("HTTP proxy", "Used for HTTP requests, e.g. http://127.0.0.1:7890", DownloadTextSetting::HttpProxy, cx).into_any_element(),
                        self.settings_download_text_row("HTTPS proxy", "Used for HTTPS requests, e.g. http://127.0.0.1:7890", DownloadTextSetting::HttpsProxy, cx).into_any_element(),
                        self.settings_download_text_row("No proxy", "Comma-separated hosts or CIDRs that bypass the proxy.", DownloadTextSetting::NoProxy, cx).into_any_element(),
                        settings_group_heading(&self.settings.language, "TLS", "Optional custom certificate authority for HTTPS connections."),
                        self.settings_download_text_row("CA bundle", "PEM certificate bundle path; blank uses system trust.", DownloadTextSetting::CaBundle, cx).into_any_element(),
                    ]))
                    .child(settings_note(&tr(&self.settings.language, "Changes apply to the next server start. If the service is running, Apply restarts it with the new settings.")));
            }
            SettingsCategory::Security => {
                content = content
                    .child(settings_panel(vec![
                        self.settings_boolean_row("API key authentication", "Require Bearer token or x-api-key for API requests", BooleanSetting::Authentication, cx).into_any_element(),
                        self.settings_api_key_row(cx).into_any_element(),
                        settings_text_row(&tr(&self.settings.language, "Health endpoint"), &tr(&self.settings.language, "Unauthenticated"), &tr(&self.settings.language, "GET /health only; all inference, model, usage and metrics routes require the key when enabled")),
                        settings_text_row(&tr(&self.settings.language, "Transport"), "HTTP", &tr(&self.settings.language, "For access beyond a trusted local network, use a TLS reverse proxy")),
                    ]))
                    .child(settings_note(&tr(&self.settings.language, "The API key is stored in macOS Keychain, not in the settings file. LAN binding is rejected until a key exists.")));
            }
            SettingsCategory::Inference => {
                content = content.child(settings_panel(vec![
                    self.settings_number_row(
                        "Context window",
                        &self.settings_context_window,
                        NumericSetting::ContextWindow,
                        "Default text context cap · oldest prompt tokens are trimmed when exceeded",
                        cx,
                    )
                    .into_any_element(),
                    self.settings_number_row(
                        "Maximum output tokens",
                        &self.settings_max_tokens,
                        NumericSetting::MaxTokens,
                        "Default 32768 · request parameters can override",
                        cx,
                    )
                    .into_any_element(),
                    self.settings_number_row(
                        "Temperature",
                        &self.settings_temperature,
                        NumericSetting::Temperature,
                        "Sampling randomness · 0 to 2",
                        cx,
                    )
                    .into_any_element(),
                    self.settings_number_row(
                        "Top P",
                        &self.settings_top_p,
                        NumericSetting::TopP,
                        "Nucleus sampling · 0 to 1",
                        cx,
                    )
                    .into_any_element(),
                    self.settings_number_row(
                        "Top K",
                        &self.settings_top_k,
                        NumericSetting::TopK,
                        "Candidate limit · 0 disables Top K",
                        cx,
                    )
                    .into_any_element(),
                    self.settings_number_row(
                        "Repetition penalty",
                        &self.settings_repetition_penalty,
                        NumericSetting::RepetitionPenalty,
                        "Penalty for repeated tokens · 1 disables penalty",
                        cx,
                    )
                    .into_any_element(),
                ]));
            }
            SettingsCategory::Startup => {
                content = content.child(settings_panel(vec![
                    self.settings_boolean_row("Start API service when MiniMLX opens", "Starts the configured model service when the desktop app launches; does not add a macOS login item", BooleanSetting::StartService, cx).into_any_element(),
                ]));
            }
            SettingsCategory::Storage => {
                let mut rows = vec![
                    self.settings_path_row("Base path", "Stores runtime data; models use the MiniMLX library and explicitly added external folders", cx).into_any_element(),
                ];
                for (index, directory) in self.settings.model_directories.iter().enumerate() {
                    let mut row = div()
                        .flex()
                        .items_center()
                        .justify_between()
                        .min_h(px(40.0))
                        .border_b_1()
                        .border_color(rgb(BORDER))
                        .child(
                            div()
                                .flex_1()
                                .text_size(px(14.0))
                                .text_color(rgb(TEXT))
                                .child(format!(
                                    "{}  {}",
                                    tr(
                                        &self.settings.language,
                                        if index == 0 {
                                            "MiniMLX library · download destination"
                                        } else {
                                            "External folder · referenced in place"
                                        },
                                    ),
                                    directory
                                )),
                        );
                    if index > 0 {
                        row = row.child(
                            app_button(SharedString::from(format!(
                                "remove-model-directory-{index}"
                            )))
                            .label(tr(&self.settings.language, "Remove"))
                            .text()
                            .on_click(cx.listener(
                                move |this, _, _, cx| {
                                    if index < this.settings.model_directories.len() {
                                        this.settings.model_directories.remove(index);
                                        this.settings_dirty = true;
                                        this.settings_message.clear();
                                    }
                                    cx.notify();
                                },
                            )),
                        );
                    }
                    rows.push(row.into_any_element());
                }
                content = content
                    .child(Self::section_title(&tr(&self.settings.language, "MiniMLX model library"), None))
                    .child(settings_panel(rows))
                    .child(settings_note(&tr(&self.settings.language, "The managed MiniMLX folder is the download destination. Only explicitly added external folders are searched; .cache directories are never scanned.")))
                    .child(Self::section_title(&tr(&self.settings.language, "External model folders"), None))
                    .child(self.settings_directory_input(cx))
                    .child(self.settings_directory_suggestions(cx))
                    .child(settings_note(&tr(&self.settings.language, "External folders are referenced in place; files are not copied or moved.")));
            }
            SettingsCategory::Logs => {
                content = content.child(settings_panel(vec![
                    self.settings_log_level_row().into_any_element(),
                    self.settings_boolean_row("Record usage history", "Store hourly per-model prompt and generated token totals for 400 days; disabling keeps existing history", BooleanSetting::UsageHistory, cx).into_any_element(),
                    settings_text_row(&tr(&self.settings.language, "History database"), &format!("{}/usage.sqlite3", self.settings_data_path), &tr(&self.settings.language, "SQLite · aggregate token totals only, not prompt or response text")),
                    settings_text_row(&tr(&self.settings.language, "Service logs"), &tr(&self.settings.language, "MiniMLX Run log"), &tr(&self.settings.language, "API process output is inherited by the app's launch environment")),
                ]));
            }
        }
        content.into_any_element()
    }
}

impl Render for MiniMlxApp {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let benchmark_models = self
            .models
            .iter()
            .filter(|model| !is_speculative_draft_entry(model))
            .map(|model| BenchmarkModelOption {
                title: model.name.clone(),
                path: model.path.clone(),
            })
            .collect::<Vec<_>>();
        let benchmark_paths = benchmark_models
            .iter()
            .map(|model| model.path.clone())
            .collect::<Vec<_>>();
        if benchmark_paths != self.benchmark_dropdown_paths {
            self.benchmark_dropdown_paths = benchmark_paths;
            let selected_model = self.selected_model.clone();
            self.benchmark_model_dropdown.update(cx, |dropdown, cx| {
                dropdown.set_items(benchmark_models, window, cx);
                dropdown.set_selected_value(&selected_model, window, cx);
            });
        }
        let page_content = match self.page {
            Page::Overview => self.overview(cx).into_any_element(),
            Page::Models => self.models_page(cx).into_any_element(),
            Page::ModelConfiguration => self.model_configuration_page(cx).into_any_element(),
            Page::Performance => self.performance_page(window, cx).into_any_element(),
            Page::Api => self.api_page(cx).into_any_element(),
            Page::Settings => self.settings_page(cx).into_any_element(),
        };
        let modal_layer = Root::render_modal_layer(window, cx);

        div()
            .flex()
            .size_full()
            .bg(rgb(CANVAS))
            .text_color(rgb(TEXT))
            .child(self.sidebar(cx))
            .child(
                div()
                    .flex()
                    .flex_col()
                    .flex_1()
                    .min_w(px(0.0))
                    .min_h(px(0.0))
                    .child(self.toolbar())
                    .child(page_content)
                    .when(self.page == Page::Performance, |container| {
                        let running = self.benchmark_running || self.operation.ends_with('…');
                        let failed =
                            self.operation.contains("failed") || self.operation.contains("失败");
                        let (status, status_color) = if running {
                            ("Running", 0x2563EB)
                        } else if failed {
                            ("Needs attention", 0xDC2626)
                        } else {
                            ("Idle", 0x059669)
                        };
                        container.child(
                            div()
                                .flex()
                                .items_center()
                                .justify_between()
                                .h(px(30.0))
                                .px_5()
                                .bg(rgb(WHITE))
                                .border_t_1()
                                .border_color(rgb(BORDER))
                                .text_size(px(11.0))
                                .text_color(rgb(MUTED))
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap_4()
                                        .child(format!(
                                            "MiniMLX Core v{}",
                                            env!("CARGO_PKG_VERSION")
                                        ))
                                        .child(tr(&self.settings.language, "Metal API active")),
                                )
                                .child(
                                    div()
                                        .flex()
                                        .items_center()
                                        .gap_4()
                                        .child(format!(
                                            "Temperature: {}",
                                            self.settings_temperature
                                        ))
                                        .child(format!("Top-p: {}", self.settings_top_p))
                                        .child(div().text_color(rgb(status_color)).child(format!(
                                            "● {}",
                                            tr(&self.settings.language, status)
                                        ))),
                                ),
                        )
                    }),
            )
            .when_some(modal_layer, |element, modal_layer| {
                element.child(modal_layer)
            })
    }
}

impl Focusable for MiniMlxApp {
    fn focus_handle(&self, _: &App) -> FocusHandle {
        self.focus_handle.clone()
    }
}

#[allow(dead_code)]
fn table_header(columns: &[&str]) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .justify_between()
        .h(px(36.0))
        .px_4()
        .bg(rgb(0xF8FAFC))
        .border_b_1()
        .border_color(rgb(BORDER))
        .children(columns.iter().map(|column| {
            div()
                .flex_1()
                .text_size(px(14.0))
                .font_weight(FontWeight::MEDIUM)
                .text_color(rgb(SECONDARY))
                .child((*column).to_string())
        }))
}

fn config_row(language: &str, label: &str, value: &str) -> impl IntoElement {
    div()
        .flex()
        .items_center()
        .justify_between()
        .min_h(px(42.0))
        .py_2()
        .border_b_1()
        .border_color(rgb(BORDER))
        .child(
            div()
                .text_size(px(14.0))
                .text_color(rgb(SECONDARY))
                .child(tr(language, label)),
        )
        .child(
            div()
                .max_w(px(520.0))
                .text_size(px(14.0))
                .text_color(rgb(TEXT))
                .child(tr(language, value)),
        )
}

fn metric_label(label: &str) -> impl IntoElement {
    div()
        .text_size(px(14.0))
        .font_weight(FontWeight::MEDIUM)
        .text_color(rgb(SECONDARY))
        .child(label.to_string())
}

fn settings_panel(rows: Vec<AnyElement>) -> AnyElement {
    div()
        .flex()
        .flex_col()
        .px_3()
        .bg(rgb(SURFACE))
        .border_1()
        .border_color(rgb(BORDER))
        .rounded_md()
        .children(rows)
        .into_any_element()
}

fn model_info_item(language: &str, label: &str, value: &str) -> AnyElement {
    div()
        .flex_1()
        .min_w(px(0.0))
        .flex()
        .flex_col()
        .gap(px(5.0))
        .child(
            div()
                .text_size(px(13.0))
                .text_color(rgb(MUTED))
                .child(tr(language, label)),
        )
        .child(
            div()
                .w_full()
                .text_size(px(15.0))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(rgb(TEXT))
                .when(label == "Architecture", |value| {
                    value.font_family("Menlo")
                })
                .truncate()
                .child(value.to_string()),
        )
        .into_any_element()
}

fn model_configuration_panel(rows: Vec<AnyElement>) -> AnyElement {
    div()
        .flex()
        .flex_col()
        .px_5()
        .pb_4()
        .bg(rgb(SURFACE))
        .border_1()
        .border_color(rgb(BORDER))
        .rounded_lg()
        .children(rows)
        .into_any_element()
}

fn model_configuration_heading(
    language: &str,
    title: &str,
    detail: &str,
    icon: gpui_component::IconName,
) -> AnyElement {
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap_4()
        .min_h(px(48.0))
        .border_b_1()
        .border_color(rgb(0xEDF0F5))
        .pt_2()
        .pb_3()
        .child(
            div()
                .flex()
                .items_center()
                .gap_2()
                .child(
                    gpui_component::Icon::new(icon)
                        .with_size(gpui_component::Size::Small)
                        .text_color(rgb(LIME)),
                )
                .child(
                    div()
                        .text_size(px(16.0))
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(rgb(TEXT))
                        .child(tr(language, title)),
                ),
        )
        .child(
            div()
                .text_size(px(12.0))
                .text_color(rgb(MUTED))
                .child(tr(language, detail)),
        )
        .into_any_element()
}

fn settings_group_heading(language: &str, title: &str, detail: &str) -> AnyElement {
    div()
        .flex()
        .flex_col()
        .gap_1()
        .pt_3()
        .pb_2()
        .child(
            div()
                .text_size(px(13.0))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(rgb(SECONDARY))
                .child(tr(language, title)),
        )
        .child(
            div()
                .text_size(px(12.0))
                .text_color(rgb(MUTED))
                .child(tr(language, detail)),
        )
        .into_any_element()
}

fn settings_text_row(label: &str, value: &str, detail: &str) -> AnyElement {
    div()
        .flex()
        .items_center()
        .justify_between()
        .gap_4()
        .min_h(px(62.0))
        .py_2()
        .border_b_1()
        .border_color(rgb(BORDER))
        .child(
            div()
                .flex()
                .flex_col()
                .gap_1()
                .child(
                    div()
                        .text_size(px(13.0))
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(rgb(TEXT))
                        .child(label.to_string()),
                )
                .child(
                    div()
                        .text_size(px(13.0))
                        .text_color(rgb(MUTED))
                        .child(detail.to_string()),
                ),
        )
        .child(
            div()
                .max_w(px(360.0))
                .text_size(px(14.0))
                .text_color(rgb(SECONDARY))
                .child(value.to_string()),
        )
        .into_any_element()
}

fn settings_note(text: &str) -> AnyElement {
    div()
        .px_3()
        .py_3()
        .rounded_sm()
        .bg(rgb(LIME_PALE))
        .text_size(px(14.0))
        .text_color(rgb(SECONDARY))
        .child(text.to_string())
        .into_any_element()
}

fn stat_card(label: &str, value: impl Into<String>, detail: &str) -> impl IntoElement {
    div()
        .flex_1()
        .min_w(px(0.0))
        .min_h(px(82.0))
        .flex()
        .flex_col()
        .gap_1()
        .px_3()
        .py_2()
        .bg(rgb(0xF5F8FC))
        .border_1()
        .border_color(rgb(BORDER))
        .rounded_md()
        .child(metric_label(label))
        .child(
            div()
                .text_size(px(19.0))
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(rgb(if label == "Avg. generation" {
                    OLIVE
                } else {
                    TEXT
                }))
                .child(value.into()),
        )
        .child(
            div()
                .text_size(px(13.0))
                .text_color(rgb(MUTED))
                .child(detail.to_string()),
        )
}

fn stat_count(metrics: &Value, key: &str) -> String {
    metrics[key]
        .as_u64()
        .map(|value| value.to_string())
        .unwrap_or_else(|| "—".into())
}

fn stat_rate(metrics: &Value, key: &str) -> String {
    metrics[key]
        .as_f64()
        .map(|value| format!("{value:.1} tok/s"))
        .unwrap_or_else(|| "—".into())
}

fn format_uptime(value: &Value) -> String {
    let Some(seconds) = value.as_u64() else {
        return "—".into();
    };
    let hours = seconds / 3600;
    let minutes = (seconds % 3600) / 60;
    if hours > 0 {
        format!("{hours}h {minutes}m")
    } else {
        format!("{minutes}m {}s", seconds % 60)
    }
}

fn format_relative_time(timestamp: u64) -> String {
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let elapsed = now.saturating_sub(timestamp);
    if elapsed < 60 {
        format!("{elapsed}s ago")
    } else if elapsed < 3600 {
        format!("{}m ago", elapsed / 60)
    } else if elapsed < 86_400 {
        format!("{}h ago", elapsed / 3600)
    } else {
        format!("{}d ago", elapsed / 86_400)
    }
}

fn format_bytes(bytes: u64) -> String {
    if bytes >= 1024 * 1024 * 1024 {
        format!("{:.1} GB", bytes as f64 / (1024.0 * 1024.0 * 1024.0))
    } else if bytes >= 1024 * 1024 {
        format!("{:.0} MB", bytes as f64 / (1024.0 * 1024.0))
    } else {
        format!("{bytes} B")
    }
}

fn format_download_speed(bytes_per_second: f64) -> String {
    let speed = bytes_per_second.max(0.0);
    if speed >= 1024.0 * 1024.0 {
        format!("{:.1} MB/s", speed / (1024.0 * 1024.0))
    } else {
        format!("{:.1} KB/s", speed / 1024.0)
    }
}

fn format_model_parameters(parameters: u64) -> String {
    if parameters >= 1_000_000_000 {
        format!("{:.1}B", parameters as f64 / 1_000_000_000.0)
    } else if parameters >= 1_000_000 {
        format!("{:.1}M", parameters as f64 / 1_000_000.0)
    } else {
        format!("{parameters} params")
    }
}

fn format_compact_count(count: u64) -> String {
    if count >= 1_000_000 {
        format!("{:.1}M", count as f64 / 1_000_000.0)
    } else if count >= 1_000 {
        format!("{:.1}K", count as f64 / 1_000.0)
    } else {
        count.to_string()
    }
}

fn action_button(
    label: &str,
    primary: bool,
    cx: &mut Context<MiniMlxApp>,
    target: Page,
) -> impl IntoElement {
    let button = app_button(SharedString::from(format!("action-{label}"))).label(label.to_string());
    let button = if primary { button.primary() } else { button };
    button.on_click(cx.listener(move |this, _, _, cx| {
        this.page = target;
        cx.notify();
    }))
}

fn app_button(id: impl Into<gpui::ElementId>) -> Button {
    Button::new(id).small()
}

fn pastel_success_tag() -> Tag {
    Tag::custom(
        rgb(0xE2F3E8).into(),
        rgb(0x265F3A).into(),
        rgb(0xC6E6D0).into(),
    )
    .small()
}

fn pastel_warning_tag() -> Tag {
    Tag::custom(
        rgb(0xFFF0C2).into(),
        rgb(0x674D00).into(),
        rgb(0xF0D98C).into(),
    )
    .small()
}

fn benchmark_model_button(
    model_path: String,
    language: &str,
    cx: &mut Context<MiniMlxApp>,
) -> impl IntoElement {
    app_button("overview-benchmark-model")
        .label(tr(language, "Benchmark"))
        .text()
        .on_click(cx.listener(move |this, _, _, cx| {
            this.selected_model = model_path.clone();
            this.page = Page::Performance;
            cx.notify();
        }))
}

fn add_model_button(language: &str, cx: &mut Context<MiniMlxApp>) -> impl IntoElement {
        app_button("action-add-model")
        .icon(gpui_component::Icon::new(gpui_component::IconName::Plus))
        .label(tr(language, "Add Model"))
        .primary()
        .on_click(cx.listener(|this, _, _, cx| {
            this.model_tab = ModelTab::Discover;
            cx.notify();
        }))
}

fn refresh_models_button(cx: &mut Context<MiniMlxApp>, language: &str) -> impl IntoElement {
    app_button("refresh-model-list")
        .label(tr(language, "Refresh"))
        .on_click(cx.listener(|this, _, _, cx| {
            if this.operation.ends_with('…') {
                return;
            }
            this.operation = tr(&this.settings.language, "Refreshing model inventory…");
            let backend = this.backend.clone();
            let task = cx.background_executor().spawn(async move {
                backend
                    .lock()
                    .map(|backend| (backend.models(), backend.incomplete_downloads()))
                    .map_err(|_| "runtime lock poisoned".to_string())
            });
            let entity = cx.entity().downgrade();
            cx.spawn(async move |_, cx| {
                let result = task.await;
                let _ = entity.update(cx, |this, cx| {
                    match result {
                        Ok((models, incomplete)) => {
                            this.models = models;
                            this.incomplete_downloads = incomplete;
                            this.operation = format!("Found {} local model(s)", this.models.len());
                        }
                        Err(error) => this.operation = format!("Inventory refresh failed: {error}"),
                    }
                    cx.notify();
                });
            })
            .detach();
        }))
}

pub fn run(model_dir: String, root: PathBuf, python: PathBuf) {
    let machine_label = detect_machine_label();
    Application::new()
        .with_assets(MiniMlxAssets)
        .run(move |cx: &mut App| {
            gpui_component::init(cx);
            // Keep GPUI Component controls aligned with MiniMLX's blue accent palette.
            let theme = Theme::global_mut(cx);
            theme.primary = rgb(0x2563EB).into();
            theme.primary_hover = rgb(0x1D4ED8).into();
            theme.primary_active = rgb(0x1E40AF).into();
            theme.primary_foreground = rgb(0xFFFFFF).into();
            theme.secondary_hover = rgb(0xDBEAFE).into();
            theme.secondary_active = rgb(0xDBEAFE).into();
            theme.accent = rgb(0xEFF6FF).into();
            theme.accent_foreground = rgb(0x1D4ED8).into();
            theme.sidebar_accent = rgb(0xEFF6FF).into();
            theme.sidebar_accent_foreground = rgb(0x1D4ED8).into();
            theme.sidebar_primary = rgb(0x2563EB).into();
            theme.sidebar_primary_foreground = rgb(0xFFFFFF).into();
            theme.link = rgb(0x2563EB).into();
            theme.link_hover = rgb(0x1D4ED8).into();
            theme.link_active = rgb(0x1E40AF).into();
            theme.ring = rgb(0x93C5FD).into();
            theme.selection = rgb(0xDBEAFE).into();
            let bounds = Bounds::centered(None, size(px(1480.0), px(920.0)), cx);
            cx.open_window(
                WindowOptions {
                    window_bounds: Some(WindowBounds::Windowed(bounds)),
                    window_min_size: Some(size(px(1180.0), px(700.0))),
                    ..Default::default()
                },
                |window, cx| {
                    let app = cx.new(|cx| {
                        MiniMlxApp::new(model_dir, root, python, machine_label, window, cx)
                    });
                    cx.new(|cx| Root::new(app.into(), window, cx))
                },
            )
            .expect("failed to open MiniMLX window");
            cx.activate(true);
        });
}

struct MiniMlxAssets;

impl AssetSource for MiniMlxAssets {
    fn load(&self, path: &str) -> gpui::Result<Option<Cow<'static, [u8]>>> {
        let bytes = match path {
            "icons/layout-dashboard.svg" => {
                include_bytes!("../assets/icons/layout-dashboard.svg").as_slice()
            }
            "icons/box.svg" => include_bytes!("../assets/icons/box.svg").as_slice(),
            "icons/blocks.svg" => include_bytes!("../assets/icons/blocks.svg").as_slice(),
            "icons/network.svg" => include_bytes!("../assets/icons/network.svg").as_slice(),
            "icons/settings.svg" => include_bytes!("../assets/icons/settings.svg").as_slice(),
            "icons/chevron-left.svg" => {
                include_bytes!("../assets/icons/chevron-left.svg").as_slice()
            }
            "icons/pause.svg" => include_bytes!("../assets/icons/pause.svg").as_slice(),
            "icons/trash-2.svg" => include_bytes!("../assets/icons/trash-2.svg").as_slice(),
            "icons/x.svg" => include_bytes!("../assets/icons/x.svg").as_slice(),
            // GPUI Component's IconName set follows Lucide's SVG assets. The
            // component resolves these through the application's AssetSource,
            // so each icon used by this app must be registered here.
            "icons/a-large-small.svg" => {
                include_bytes!("../assets/icons/a-large-small.svg").as_slice()
            }
            "icons/bot.svg" => include_bytes!("../assets/icons/bot.svg").as_slice(),
            "icons/chevron-down.svg" => {
                include_bytes!("../assets/icons/chevron-down.svg").as_slice()
            }
            "icons/chevron-up.svg" => {
                include_bytes!("../assets/icons/chevron-up.svg").as_slice()
            }
            "icons/copy.svg" => include_bytes!("../assets/icons/copy.svg").as_slice(),
            "icons/external-link.svg" => {
                include_bytes!("../assets/icons/external-link.svg").as_slice()
            }
            "icons/plus.svg" => include_bytes!("../assets/icons/plus.svg").as_slice(),
            "icons/search.svg" => include_bytes!("../assets/icons/search.svg").as_slice(),
            "icons/settings-2.svg" => {
                include_bytes!("../assets/icons/settings-2.svg").as_slice()
            }
            _ => return Ok(None),
        };
        Ok(Some(Cow::Borrowed(bytes)))
    }

    fn list(&self, path: &str) -> gpui::Result<Vec<SharedString>> {
        if path != "icons" {
            return Ok(Vec::new());
        }
        Ok([
            "layout-dashboard.svg",
            "box.svg",
            "blocks.svg",
            "network.svg",
            "settings.svg",
            "chevron-left.svg",
            "pause.svg",
            "trash-2.svg",
            "x.svg",
            "a-large-small.svg",
            "bot.svg",
            "chevron-down.svg",
            "chevron-up.svg",
            "copy.svg",
            "external-link.svg",
            "plus.svg",
            "search.svg",
            "settings-2.svg",
        ]
        .into_iter()
        .map(SharedString::from)
        .collect())
    }
}

#[cfg(test)]
mod asset_tests {
    use super::*;

    #[test]
    fn temperature_accepts_any_finite_non_negative_value() {
        for value in [0.0, 0.7, 2.0, 2.5, 10.0, f32::MAX] {
            assert!(valid_temperature(value), "should accept {value}");
        }
        for value in [-0.01, f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            assert!(!valid_temperature(value), "should reject {value}");
        }
    }

    #[test]
    fn download_action_icons_are_available_from_the_asset_source() {
        let assets = MiniMlxAssets;
        for icon in [
            "icons/pause.svg",
            "icons/trash-2.svg",
            "icons/x.svg",
            "icons/a-large-small.svg",
            "icons/bot.svg",
            "icons/chevron-down.svg",
            "icons/chevron-up.svg",
            "icons/copy.svg",
            "icons/external-link.svg",
            "icons/plus.svg",
            "icons/search.svg",
            "icons/settings-2.svg",
        ] {
            assert!(
                assets.load(icon).unwrap().is_some(),
                "missing asset: {icon}"
            );
        }
        let listed = assets.list("icons").unwrap();
        assert!(listed.iter().any(|name| name.as_ref() == "pause.svg"));
        assert!(listed.iter().any(|name| name.as_ref() == "trash-2.svg"));
        assert!(listed.iter().any(|name| name.as_ref() == "x.svg"));
        for icon in [
            "a-large-small.svg",
            "bot.svg",
            "chevron-down.svg",
            "chevron-up.svg",
            "copy.svg",
            "external-link.svg",
            "plus.svg",
            "search.svg",
            "settings-2.svg",
        ] {
            assert!(listed.iter().any(|name| name.as_ref() == icon));
        }
    }

    #[test]
    fn download_speed_is_shown_in_readable_units() {
        assert_eq!(format_download_speed(571_688.0), "558.3 KB/s");
        assert_eq!(format_download_speed(2_097_152.0), "2.0 MB/s");
        assert_eq!(format_download_speed(0.0), "0.0 KB/s");
    }

    #[test]
    fn benchmark_report_extracts_context_rows_and_raw_log() {
        let report = "Reports/run.md\n\ncontext=1024 TTFT=545.6 ms TPOT=15.43 ms PP_TPS=1876.7 TG_TPS=65.32 E2E=2571.3 ms generated=128/128\n  memory active=3.993 GiB peak=4.754 GiB\n";
        let rows = context_benchmark_rows(report);
        assert_eq!(rows.len(), 1);
        assert_eq!(rows[0].context, "1024");
        assert_eq!(rows[0].ttft, "545.6");
        assert_eq!(rows[0].e2e_seconds, "2.6");
        assert_eq!(rows[0].peak_memory, "4.754 GiB");
        assert!(benchmark_log(report).starts_with("context=1024"));
    }

    #[test]
    fn benchmark_report_reads_saved_markdown_fence_as_log() {
        let report = "# Report\n\n## Raw benchmark output\n\n```text\ncontext=4096 TTFT=10 ms TPOT=2 ms PP_TPS=100 TG_TPS=50 E2E=20 ms\n```\n";
        assert!(benchmark_log(report).starts_with("context=4096"));
        assert_eq!(context_benchmark_rows(report).len(), 1);
    }

    #[test]
    fn benchmark_report_identifies_active_speculative_draft() {
        let mtp = "- MTP draft: `qwen-mtp`\n- DFlash draft: `disabled`";
        assert_eq!(benchmark_speculative_draft(mtp), Some(("MTP".into(), "qwen-mtp".into())));
        let dflash = "- MTP draft: `disabled`\n- DFlash draft: `qwen-dflash2`";
        assert_eq!(benchmark_speculative_draft(dflash), Some(("DFlash".into(), "qwen-dflash2".into())));
    }

    #[test]
    fn context_labels_use_compact_k_units() {
        assert_eq!(format_context_label("1024"), "1K");
        assert_eq!(format_context_label("4096"), "4K");
        assert_eq!(format_context_label("8192"), "8K");
        assert_eq!(format_context_label("16384"), "16K");
        assert_eq!(format_context_label("1500"), "1500");
    }

    #[test]
    fn benchmark_table_exports_displayed_values_as_csv() {
        let report = "# Report\n\n## Raw benchmark output\n\n```text\ncontext=1024 TTFT=545.6 ms TPOT=15.43 ms PP_TPS=1876.7 TG_TPS=65.32 E2E=2571.3 ms generated=128/128\n  memory active=3.993 GiB peak=4.754 GiB\n```\n";
        let csv = context_benchmark_csv(report).unwrap();
        assert!(csv.starts_with("Context,TTFT (ms),TPOT (ms),Prompt TPS (tok/s)"));
        assert!(csv.contains("1K,545.6,15.43,1876.7,65.32,2.6,4.754 GiB"));
    }

    #[test]
    fn model_inference_settings_match_equivalent_path_spellings() {
        let unique = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let root = std::env::temp_dir().join(format!("minimlx-settings-path-{unique}"));
        let model = root.join("model");
        fs::create_dir_all(root.join("alias")).unwrap();
        fs::create_dir_all(&model).unwrap();

        let alias = root.join("alias/../model").display().to_string();
        let canonical = model.canonicalize().unwrap().display().to_string();
        let mut settings = DesktopSettings::default();
        let mut options = ModelInferenceOptions::default();
        options.mtp_draft_model = Some("/models/qwen-mtp".into());
        settings.model_inference.insert(alias, options);

        let resolved = model_inference_options_for_path(&settings, &canonical).unwrap();
        assert_eq!(
            resolved.mtp_draft_model.as_deref(),
            Some("/models/qwen-mtp")
        );
        assert!(model_settings_saved_message(resolved).contains("MTP enabled"));

        fs::remove_dir_all(root).unwrap();
    }
}

fn detect_machine_label() -> String {
    let chip = Command::new("sysctl")
        .args(["-n", "machdep.cpu.brand_string"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .map(|output| String::from_utf8_lossy(&output.stdout).trim().to_string())
        .filter(|value| !value.is_empty())
        .unwrap_or_else(|| "Apple Silicon".to_string());
    let memory_gb = Command::new("sysctl")
        .args(["-n", "hw.memsize"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| {
            String::from_utf8_lossy(&output.stdout)
                .trim()
                .parse::<u64>()
                .ok()
        })
        .map(|bytes| bytes / 1024 / 1024 / 1024);
    match memory_gb {
        Some(memory_gb) => format!("{chip} · {memory_gb} GB"),
        None => chip,
    }
}
