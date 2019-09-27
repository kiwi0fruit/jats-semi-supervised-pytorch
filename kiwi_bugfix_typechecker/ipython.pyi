from typing import Any
from IPython.display import DisplayHandle


def display(*objs: Any, include=None, exclude=None, metadata=None, transient=None,
            display_id=None, **kwargs) -> DisplayHandle: ...
