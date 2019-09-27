#!/bin/bash
export PYTHONUTF8=1
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
merge_pkg () {
    pkg="$1"
    stubsdir="$2"
    here="$PWD"
    moddir="$(python -c "import $pkg as m; import os.path.dirname as d; print(d(m.__file__))")"
    if [[ "$OSTYPE" == "msys" ]]; then
        moddir="$(cygpath "$moddir")"; fi
    cd "$moddir"
    mkdir -p "$stubsdir/$pkg"
    find . -name '*.pyi' -exec cp --parents \{\} "$stubsdir/$pkg" \;
    cd "$here"
}
cross_path () {
    in_path="$1"
    if [[ "$OSTYPE" == "msys" ]]; then
        in_path="$(cygpath -w "${in_path}")"
        in_path="${in_path//[\\]/\\\\}"; fi
    echo "${in_path}"
}


stubs="$(dirname "$PWD")/stubs"


printf "[mypy]
ignore_missing_imports = True
namespace_packages = True
mypy_path = $(cross_path "$stubs")

# [mypy-torch.*]
# ignore_errors = True
" > ./mypy.ini


if [ -d "$stubs" ]; then rm -Rf "$stubs"; fi
mkdir -p "$stubs"


# stubgen -p torch -o "$stubs"
# merge_pkg torch "$stubs"
# rm "$stubs/torch/_C.pyi" || true

stubgen -p normalizing_flows_typed -o "$stubs"
stubgen -p beta_tcvae_typed -o "$stubs"
stubgen -p semi_supervised_typed -o "$stubs"
stubgen -p vae -o "$stubs"
stubgen -p socionics_db -o "$stubs"
stubgen -p jats_vae -o "$stubs"
stubgen -p jats_display -o "$stubs"
stubgen ./ready.py -o "$stubs"
