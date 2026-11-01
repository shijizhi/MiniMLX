#!/usr/bin/env python3
"""Rewrite copied Python framework load paths to loader-relative paths."""
import os
import subprocess
import sys
from pathlib import Path

runtime = Path(sys.argv[1]).resolve()
framework = sys.argv[2].rstrip("/")
for path in runtime.rglob("*"):
    if path.is_symlink() or not path.is_file():
        continue
    if path.name != "Python" and path.name != "python3.13" and path.suffix not in (".so", ".dylib"):
        continue
    result = subprocess.run(["otool", "-L", str(path)], capture_output=True, text=True)
    if result.returncode:
        continue
    dependencies = set()
    for line in result.stdout.splitlines()[1:]:
        dependency = line.strip().split(" (", 1)[0]
        if dependency.startswith(framework + "/"):
            dependencies.add(dependency)
    for dependency in dependencies:
        target = runtime / dependency[len(framework) + 1:]
        if not target.exists():
            raise SystemExit(f"Missing bundled dependency: {dependency} referenced by {path}")
        relative = os.path.relpath(target, path.parent)
        subprocess.run(["install_name_tool", "-change", dependency, "@loader_path/" + relative, str(path)], check=True)
    if path.name == "Python":
        subprocess.run(["install_name_tool", "-id", "@rpath/Python", str(path)], check=True)
    if dependencies or path.name == "Python":
        subprocess.run(["codesign", "--force", "--sign", "-", str(path)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
