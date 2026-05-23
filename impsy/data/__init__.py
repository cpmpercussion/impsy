"""Bundled data resources shipped inside the impsy wheel.

Currently holds the default config template used by `impsy init` and the
webui's first-run auto-create path.
"""

from importlib.resources import files


def default_config_template() -> str | None:
    """Return the bundled default.toml template as text, or None if missing."""
    try:
        return files(__name__).joinpath("default.toml").read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
