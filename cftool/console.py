from typing import Any
from typing import List
from typing import Optional

from rich.prompt import Prompt
from rich.status import Status
from rich.console import Console

from .misc import shallow_copy_dict


_console = Console()


def log(msg: str, *args: Any, **kwargs: Any) -> None:
    _console.log(msg, *args, **kwargs)


def debug(msg: str, *args: Any, prefix: str = "", **kwargs: Any) -> None:
    log(f"[grey42]{prefix}{msg}[/grey42]", *args, **kwargs)


def warn(msg: str, *args: Any, prefix: str = "Warning: ", **kwargs: Any) -> None:
    log(f"[yellow]{prefix}{msg}[/yellow]", *args, **kwargs)


def deprecated(msg: str, *args: Any, **kwargs: Any) -> None:
    warn(msg, *args, prefix="DeprecationWarning: ", **kwargs)


def error(msg: str, *args: Any, prefix: str = "Error: ", **kwargs: Any) -> None:
    log(f"[red]{prefix}{msg}[/red]", *args, **kwargs)


def print(msg: str, *args: Any, **kwargs: Any) -> None:
    _console.print(msg, *args, **kwargs)


def rule(title: str, **kwargs: Any) -> None:
    _console.rule(title, **kwargs)


def ask(
    question: str,
    choices: Optional[List[str]] = None,
    *,
    default: Optional[str] = None,
    **kwargs: Any,
) -> str:
    kwargs = shallow_copy_dict(kwargs)
    kwargs["choices"] = choices
    if default is not None:
        kwargs["default"] = default
    return Prompt.ask(question, **kwargs)


def status(msg: str, **kwargs: Any) -> Status:
    return _console.status(msg, **kwargs)
