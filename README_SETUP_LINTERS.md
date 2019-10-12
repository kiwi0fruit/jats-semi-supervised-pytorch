Linting in PyCharm Community Edition and Visual Studio Code
====================================
I simultaneously use these two IDE as either one is not yet ideal.


PyCharm linting
====================================

Works out of the box. You only need to ignore errors like present in this repo as I did.

PEP8 options that I disable in PyCharm but mostly follow:

* N802 - function name should be lowercase.
  Sometimes prevents from using the best readable name otherwise is good.
* N803 - argument name should be lowercase.
  Sometimes prevents from using the best readable name otherwise is good.
* N806 - variable in function should be lowercase.
  Sometimes prevents from using the best readable name otherwise is good.
* E302 - expected 2 blank lines, found 1.
  Sometimes prevents from writing neat functions definitions otherwise is good.
* E305 - expected 2 blank lines after end of function or class.
  Sometimes prevents from writing neat functions definitions otherwise is good.
* E306 - expected 1 blank line before a nested definition.
  Sometimes prevents from writing neat functions definitions otherwise is good.
* E252 - missing whitespace around parameter equals.
  Useless.
* E704 - Multiple statements on one line (def).
  Useless.
* W503, W504 - Line break occurred before a binary operator, Line break occurred after a binary operator.
  Useless.
* E501 - line too long (X > 80 characters).
  Should be used with modified checker like in PyCharm: (X > 120 characters).
* E741 - ambiguous variable name.
  Is buggy with UTF-8 names otherwise is good.
* E402 - module level import not at top of file.
  Should be disabled via comments `import this  # noqa: E402`.


mypy
====================================

I was only able to get working mypy GUI for Windows via crossplatform
[Mypy extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=matangover.mypy)
([source](https://github.com/matangover/mypy-vscode)).

But it needs some additional installation. For conda:

batch or git-bash:

```bash
conda create -n mypyls -c defaults -c conda-forge "python>=3.7" future mypy python-jsonrpc-server typed-ast mypy_extensions exec-wrappers
activate mypyls
# pip install "https://github.com/matangover/mypyls/archive/master.zip#egg=mypyls[default-mypy]"
pip install git+"https://github.com/kiwi0fruit/mypyls#egg=mypyls[default-mypy]" || exi
```

git-bash only:

```bash
exec=mypyls
execdir="$(dirname "$(type -p "$exec")")"
create-wrappers -t conda -b "$execdir" -f "$exec" -d "$(dirname "$execdir/wrap/$exec")" --conda-env-dir "$(dirname "$execdir")"
```

Then set mypy executable in Visual Studio Code to `<...>\Miniconda3\envs\mypyls\Scripts\wrap\mypyls.bat`

This may work without activation script wrapper but with this it's the safest way.

All these packages can be also be installed right into the target environment.
Also worth noting that you would need numpy stubs:

```bash
pip install --user git+"https://github.com/numpy/numpy-stubs"
```

I installed with `--user` because I simultaneously use PyCharm and Visual Studio Code.
Hence I exclude userdata Python folder from PyCharm
environment but mypy still sees it (that's important as PyCharm uses it's own numpy stubs
and without excluding it would use two simultaneously).

**Finally and most importantly** generate `mypy.ini` and project stubs by running `mypy.sh` from activated environment with mypy.


pylint
================================================

I was able to get usable pylint GUI for Windows that can lint the whole project only in crossplatform
[Pylint plugin for PyCharm](https://plugins.jetbrains.com/plugin/11084-pylint).
So you first may want to disable pylint in the
[Python extension for Visual Studio Code](https://marketplace.visualstudio.com/items?itemName=ms-python.python)
if you have it.

In the Pylint plugin for PyCharm settings you would need to set path to pylint executable and to `.pylintrc`
in this folder that is generated via `pylint.sh`
(this workaround is needed to get the whole project linting working). After that never open Pylint plugin settings tab as
it's buggy and may require restarting PyCharm.
