#!/bin/bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
FRAMEWORK="${MINIMLX_PYTHON_FRAMEWORK:-/Library/Frameworks/Python.framework/Versions/3.13}"
SITE_PACKAGES="${MINIMLX_SITE_PACKAGES:-$ROOT/.venv/lib/python3.13/site-packages}"
OUTPUT="${MINIMLX_PACKAGE_OUTPUT:-$ROOT/dist}"
IDENTITY="${MINIMLX_SIGN_IDENTITY:--}"
VERSION="$("$ROOT/.venv/bin/python" "$ROOT/tools/version.py")"
if [[ "$IDENTITY" == "-" ]]; then
    IMAGE="$OUTPUT/MiniMLX-$VERSION-test-arm64.dmg"
else
    IMAGE="$OUTPUT/MiniMLX-$VERSION-arm64.dmg"
fi
[[ ! -e "$IMAGE" ]] || { echo "Refusing to overwrite existing image: $IMAGE" >&2; exit 1; }
WORK="$(mktemp -d "${TMPDIR:-/tmp}/minimlx-package.XXXXXXXX")"
trap 'rm -rf "$WORK"' EXIT
sign_with_retry() {
    local attempt
    for attempt in 1 2 3; do
        codesign "$@" && return 0
        [[ "$attempt" == 3 ]] || sleep 2
    done
    return 1
}
[[ "$(uname -s)" == Darwin && "$(uname -m)" == arm64 ]] || { echo 'Apple Silicon macOS is required.' >&2; exit 1; }
[[ -f "$FRAMEWORK/Python" && -d "$SITE_PACKAGES" ]] || { echo 'Python framework or site-packages not found.' >&2; exit 1; }
mkdir -p "$OUTPUT" "$WORK/python/bin" "$WORK/python/lib"
cp "$FRAMEWORK/Python" "$WORK/python/Python"
cp "$FRAMEWORK/Resources/Python.app/Contents/MacOS/Python" "$WORK/python/bin/python3.13"
cp -R "$FRAMEWORK/lib/python3.13" "$WORK/python/lib/"
cp -R "$FRAMEWORK/lib/"*.dylib "$WORK/python/lib/"
# Tk is not used by the application and otherwise pulls in a separate GUI framework.
rm -f "$WORK/python/lib/python3.13/lib-dynload/"_tkinter*.so
rm -rf "$WORK/python/lib/python3.13/site-packages"
cp -R "$SITE_PACKAGES" "$WORK/python/lib/python3.13/site-packages"
# A framework installation embeds absolute load commands. Rewrite them relative
# to each binary so the copied runtime does not depend on /Library/Frameworks.
/usr/bin/python3 "$ROOT/packaging/relocate_python.py" "$WORK/python" "$FRAMEWORK"
cat > "$WORK/python/bin/python3" <<'WRAPPER'
#!/bin/bash
RUNTIME="$(cd "$(dirname "$0")/.." && pwd)"
export PYTHONHOME="$RUNTIME"
export PYTHONNOUSERSITE=1
exec "$RUNTIME/bin/python3.13" "$@"
WRAPPER
chmod +x "$WORK/python/bin/python3"
"$WORK/python/bin/python3" -c 'import importlib.util,platform,sys,ssl; assert sys.version_info[:2] == (3,13) and platform.machine() == "arm64"; assert all(importlib.util.find_spec(x) for x in ("mlx", "mlx_vlm", "mlx_lm"))'
(cd "$ROOT" && XDG_CACHE_HOME="$WORK/cache" CLANG_MODULE_CACHE_PATH="$WORK/clang-cache" cargo build --release -p minimlx-gpui)
APP="$WORK/MiniMLX.app"
RES="$APP/Contents/Resources"
mkdir -p "$APP/Contents/MacOS" "$RES/tools" "$RES/Resources"
cp "$ROOT/target/release/minimlx-gpui" "$APP/Contents/MacOS/MiniMLX"
cp "$ROOT/crates/minimlx-gpui/assets/brand/AppIcon.icns" "$RES/AppIcon.icns"
cp -R "$WORK/python" "$RES/runtime"
cp -R "$ROOT/python" "$RES/python"
cp -R "$ROOT/Resources/BenchmarkCorpus" "$RES/Resources/BenchmarkCorpus"
cp "$ROOT/tools/benchmark_context_matrix.py" "$RES/tools/"
find "$RES/python" -type d -name __pycache__ -prune -exec rm -rf {} +
cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>CFBundleName</key><string>MiniMLX</string>
<key>CFBundleDisplayName</key><string>MiniMLX</string>
<key>CFBundleIdentifier</key><string>com.minimlx.app</string>
<key>CFBundleExecutable</key><string>MiniMLX</string>
<key>CFBundleIconFile</key><string>AppIcon</string>
<key>CFBundlePackageType</key><string>APPL</string>
<key>CFBundleShortVersionString</key><string>$VERSION</string>
<key>CFBundleVersion</key><string>$VERSION</string>
<key>NSHighResolutionCapable</key><true/>
</dict></plist>
PLIST
plutil -lint "$APP/Contents/Info.plist"
if [[ "$IDENTITY" == "-" ]]; then
    codesign --force --deep --sign - "$APP"
else
    # Sign nested Mach-O code before sealing the application bundle.
    while IFS= read -r -d '' item; do
        if file -b "$item" | grep -q 'Mach-O'; then
            sign_with_retry --force --sign "$IDENTITY" --timestamp --options runtime \
                --entitlements "$ROOT/packaging/runtime.entitlements" "$item"
        fi
    done < <(find "$RES/runtime" -type f -print0)
    sign_with_retry --force --sign "$IDENTITY" --timestamp --options runtime \
        --entitlements "$ROOT/packaging/runtime.entitlements" "$APP"
fi
codesign --verify --deep --strict "$APP"
mkdir "$WORK/dmg"
cp -R "$APP" "$WORK/dmg/"
ln -s /Applications "$WORK/dmg/Applications"
hdiutil create -quiet -volname MiniMLX -srcfolder "$WORK/dmg" -format UDZO "$IMAGE"
if [[ "$IDENTITY" != "-" ]]; then
    sign_with_retry --force --sign "$IDENTITY" --timestamp "$IMAGE"
fi
(cd "$(dirname "$IMAGE")" && shasum -a 256 "$(basename "$IMAGE")") > "$IMAGE.sha256"
echo "Image: $IMAGE"
echo 'Not notarized: this image is not ready for public release.'
