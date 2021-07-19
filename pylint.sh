#!/bin/bash
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cross_esc_path () {
    in_path="$1"
    if [[ "$OSTYPE" == "msys" ]]; then
        in_path="$(cygpath -w "${in_path}")"
        in_path="${in_path//[\\]/\\\\\\\\}"; fi
    echo "${in_path}"
}

printf "[MESSAGES CONTROL]
disable=abstract-method,attribute-defined-outside-init,duplicate-code,\
too-many-instance-attributes,too-many-branches,too-many-return-statements,\
too-many-nested-blocks,no-member,not-callable,no-name-in-module,\
import-error,missing-docstring,invalid-name,too-many-locals,\
too-many-arguments,too-few-public-methods,too-many-statements,\
superfluous-parens,bad-whitespace,bad-continuation,\
multiple-statements,pointless-string-statement,unsubscriptable-object,\
too-many-boolean-expressions,relative-beyond-top-level,too-many-ancestors,\
len-as-condition,unneeded-not,self-assigning-variable,import-outside-toplevel

[FORMAT]
max-line-length=120

[MASTER]
init-hook=\'import sys; sys.path.append(\"$(cross_esc_path "$PWD")\")\'
" > ./.pylintrc
