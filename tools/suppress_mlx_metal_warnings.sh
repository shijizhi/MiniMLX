#!/bin/zsh

set -euo pipefail

project_root="${0:A:h}/.."
cmake_file="$project_root/.build/checkouts/mlx-swift/Source/Cmlx/mlx/mlx/backend/metal/kernels/CMakeLists.txt"

if [[ ! -f "$cmake_file" ]]; then
  print -u2 "MLX checkout not found: $cmake_file"
  print -u2 "Run swift package resolve first."
  exit 1
fi

if rg -q --fixed-strings -- "-Wno-unused-variable" "$cmake_file"; then
  print "MLX Metal unused-variable warning suppression is already enabled."
  exit 0
fi

perl -0pi -e 's/(\s+-Wextra\n)/$1      -Wno-unused-variable\n/' "$cmake_file"
print "Enabled -Wno-unused-variable in MLX Metal kernel compilation."
