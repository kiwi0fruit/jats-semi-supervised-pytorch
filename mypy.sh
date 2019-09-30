#!/bin/bash
export PYTHONUTF8=1
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cross_path () {
    in_path="$1"
    if [[ "$OSTYPE" == "msys" ]]; then
        in_path="$(cygpath -w "${in_path}")"
        in_path="${in_path//[\\]/\\\\}"; fi
    echo "${in_path}"
}


printf "[mypy]
ignore_missing_imports = True
namespace_packages = True
mypy_path = $(cross_path "$PWD")

# [mypy-torch.*]
# ignore_errors = True
" > ./mypy.ini
