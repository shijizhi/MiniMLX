# MiniMLX 发布流程

本文面向维护者，描述在 Apple Silicon Mac 上手动构建和发布 GitHub Release 的流程。整个过程不依赖 CI。

## 准备

- Apple Silicon Mac、Xcode 命令行工具、Rust 工具链和可用的 Python 3.13 开发环境。
- 登录钥匙串中有可使用的 **Developer ID Application** 证书及其私钥。用 `security find-identity -v -p codesigning` 确认。
- `notarytool` 已配置钥匙串凭据。首次配置可运行 `xcrun notarytool store-credentials "MiniMLX-Notary"`，凭据由工具交互式读取并存入钥匙串，不要写入仓库。
- 确认 `LICENSE`、`NOTICE`、打包依赖及其第三方许可证和归属信息适用于本次分发。

当前打包脚本从本机 `/Library/Frameworks/Python.framework/Versions/3.13` 和项目 `.venv/lib/python3.13/site-packages` 复制运行环境，再修正框架内的绝对动态库路径。可通过 `MINIMLX_PYTHON_FRAMEWORK` 和 `MINIMLX_SITE_PACKAGES` 指定其他来源。**它不会在干净环境中重新解析或锁定全部依赖**；发布前必须检查这些目录的内容、依赖版本和许可证，避免将不需要的开发包或本机信息带入安装包。

## 1. 确定版本和发布内容

查看当前版本及一致性：

```bash
.venv/bin/python tools/version.py
cargo metadata --locked --offline --no-deps --format-version 1 >/dev/null
```

如果需要更新版本，运行 `.venv/bin/python tools/version.py 0.2.0`，其中 `0.2.0` 是示例。该命令同步 Rust、Python 和运行时版本；完整说明见 [README.md](README.md)。检查改动、运行适当测试并提交发布源码。构建时应使用与将来标签完全一致的提交，避免从含未提交改动的工作树制作正式包。

## 2. 构建并签名

将下面的占位值替换为 `security find-identity` 输出中的 Developer ID Application 身份或指纹：

```bash
MINIMLX_SIGN_IDENTITY="<Developer ID Application 身份或指纹>" bash tools/package_macos.sh
```

脚本生成 `dist/MiniMLX-<版本>-arm64.dmg` 和同名 `.sha256`，并拒绝覆盖已有 DMG。未指定签名身份时生成的是临时签名的 `-test` 包，不能用于正式发布。构建成功只表示已签名，**不代表已公证**。

## 3. 公证并附加票据

```bash
bash tools/notarize_macos.sh
```

脚本默认提交当前版本的正式 DMG，使用钥匙串配置 `MiniMLX-Notary`；其他配置可通过 `MINIMLX_NOTARY_PROFILE` 指定。也可以将 DMG 路径作为第一个参数。公证通过后，脚本附加并验证票据，最后重新生成 SHA-256。若公证失败，使用 `xcrun notarytool log <提交 ID> --keychain-profile MiniMLX-Notary` 查看 Apple 返回的诊断，修复后重新构建和提交。

## 4. 验证安装包

```bash
release_version="$(.venv/bin/python tools/version.py)"
xcrun stapler validate "dist/MiniMLX-${release_version}-arm64.dmg"
(cd dist && shasum -a 256 -c "MiniMLX-${release_version}-arm64.dmg.sha256")
```

在另一台或干净的 Apple Silicon Mac 上从 DMG 拖入 Applications，确认 Gatekeeper 接受应用，并检查界面版本号、Dock 与菜单栏、模型下载和加载、API 请求、性能测试，以及适用模型的 SSD 分页和 MTP。确认支持的最低 macOS 版本后，在发布说明中写明；不要根据构建机器的系统版本推断最低版本。

## 5. GitHub Release

以已验证的源码提交创建对应标签，例如 `v0.2.0`。在 GitHub 创建 Release，参考 [RELEASE_NOTES.md](RELEASE_NOTES.md) 编写版本说明，上传公证后的 DMG 和其 SHA-256 文件。检查下载文件与发布说明中的版本、架构和标签一致后再发布。

发布前如有代码、依赖或资源变化，应重新构建、签名、公证和验证；不要修改已公证的安装包后直接分发。
