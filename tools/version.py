#!/usr/bin/env python3
"""Check or update MiniMLX release versions (Python 3.11+)."""
import argparse
import re
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGES = ("minimlx-api", "minimlx-gpui", "minimlx-protocol", "minimlx-supervisor")


def files(root):
    return (root / "Cargo.toml", root / "pyproject.toml", root / "Cargo.lock", root / "python/minimlx_runtime/_version.py")


def validate(value):
    match = re.fullmatch(r"(0|[1-9]\d{0,3})\.(0|[1-9]\d?)\.(0|[1-9]\d?)", value)
    if not match:
        raise ValueError("version must be MAJOR.MINOR.PATCH (major <= 9999, minor/patch <= 99)")
    return value


def replace_one(text, pattern, replacement):
    result, count = re.subn(pattern, replacement, text, flags=re.MULTILINE)
    if count != 1:
        raise ValueError(f"expected one version entry, found {count}: {pattern}")
    return result


def get_versions(root):
    cargo, pyproject, lock, module = files(root)
    workspace = tomllib.loads(cargo.read_text())["workspace"]["package"]["version"]
    python = tomllib.loads(pyproject.read_text())["project"]["version"]
    packages = {p["name"]: p["version"] for p in tomllib.loads(lock.read_text())["package"] if p["name"] in PACKAGES}
    match = re.search(r'^__version__ = "([^"]+)"$', module.read_text(), re.MULTILINE)
    if set(packages) != set(PACKAGES) or not match:
        raise ValueError("missing workspace lock entries or Python runtime version")
    return workspace, python, packages, match.group(1)


def check(root=ROOT):
    workspace, python, packages, runtime = get_versions(root)
    validate(workspace)
    if any(v != workspace for v in [python, runtime, *packages.values()]):
        raise ValueError(f"inconsistent versions: workspace={workspace}, Python={python}, runtime={runtime}, lock={packages}")
    return workspace


def set_version(version, root=ROOT):
    version = validate(version)
    check(root)
    cargo, pyproject, lock, module = files(root)
    sources = [p.read_text() for p in (cargo, pyproject, lock, module)]
    sources[0] = replace_one(sources[0], r'(?s)(\[workspace\.package\][^\[]*?\nversion = ")[^"]+("\s*)', rf'\g<1>{version}\g<2>')
    sources[1] = replace_one(sources[1], r'(?s)(\[project\][^\[]*?\nversion = ")[^"]+("\s*)', rf'\g<1>{version}\g<2>')
    for name in PACKAGES:
        sources[2] = replace_one(sources[2], rf'(name = "{name}"\nversion = ")[^"]+("\n)', rf'\g<1>{version}\g<2>')
    sources[3] = replace_one(sources[3], r'^__version__ = "[^"]+"$', f'__version__ = "{version}"')
    for path, source in zip((cargo, pyproject, lock, module), sources):
        path.write_text(source)
    return check(root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("version", nargs="?", help="new release version, e.g. 0.2.0")
    args = parser.parse_args()
    try:
        print(set_version(args.version) if args.version else check())
    except (ValueError, OSError, KeyError) as error:
        print(f"version error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
