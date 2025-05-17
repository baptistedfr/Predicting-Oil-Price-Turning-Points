import logging.config
from rich.console import Console
import functools

console = Console()

LOGGING_CONFIG = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "file": {
            "format": "%(asctime)s [%(levelname)s] %(name)s:%(lineno)d – %(message)s",
            "datefmt": "%Y-%m-%d %H:%M:%S",
        }
    },
    "handlers": {
        "rich": {
            "()": "rich.logging.RichHandler",
            "console": console,
            "show_time": True,
            "show_level": True,
            "show_path": False,
            "level": "INFO",
        }
    },
    "root": {
        "handlers": ["rich"],
    },
}

def configure_logger(level: str = "DEBUG"):

    LOGGING_CONFIG["root"]["level"] = level
    LOGGING_CONFIG["handlers"]["rich"]["level"] = level
    logging.config.dictConfig(LOGGING_CONFIG)

    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)
    logging.getLogger("numba").setLevel(logging.WARNING)


def with_spinner(text: str, spinner: str = "simpleDotsScrolling"):
    """
    Decorator factory that displays a console spinner with a status message while the decorated function executes.
    Parameters
    ----------
    text : str
        A template for the status message, which can include placeholders corresponding
        to the decorated function’s arguments.
    spinner : str, optional
        The name of the spinner animation to display (default is "simpleDotsScrolling").
    Returns
    -------
    Callable[[Callable[..., T]], Callable[..., T]]
        A decorator that wraps a function so that when it is called, a status spinner
        is shown in the console with the formatted message until the function completes.
    Notes
    -----
    - Function arguments are bound to their parameter names using `inspect.signature`
      and `bind_partial`, and then applied with defaults.
    - If message formatting fails, the raw `text` is displayed.
    - Requires a `console` object with a `.status()` context manager supporting
      `spinner` and rich-style formatted text.
    """
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Merge args→kwargs via les noms de signature
            import inspect
            sig = inspect.signature(func)
            bound = sig.bind_partial(*args, **kwargs)
            bound.apply_defaults()

            try:
                message = text.format(**bound.arguments)
            except Exception:
                message = text

            with console.status(f"[bold green]{message}", spinner=spinner):
                return func(*args, **kwargs)
        return wrapper
    return decorator