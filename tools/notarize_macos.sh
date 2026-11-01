#!/bin/bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VERSION="$("$ROOT/.venv/bin/python" "$ROOT/tools/version.py")"
IMAGE="${1:-$ROOT/dist/MiniMLX-$VERSION-arm64.dmg}"
PROFILE="${MINIMLX_NOTARY_PROFILE:-MiniMLX-Notary}"
[[ -f "$IMAGE" ]] || { echo "DMG not found: $IMAGE" >&2; exit 1; }
codesign --verify --strict --verbose=2 "$IMAGE"
xcrun notarytool submit "$IMAGE" --keychain-profile "$PROFILE" --wait
xcrun stapler staple "$IMAGE"
xcrun stapler validate "$IMAGE"
(cd "$(dirname "$IMAGE")" && shasum -a 256 "$(basename "$IMAGE")") > "$IMAGE.sha256"
echo "Notarized image: $IMAGE"
