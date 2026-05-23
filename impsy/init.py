"""Workspace scaffolding command: `impsy init`."""

from pathlib import Path

import click

from impsy.data import default_config_template

WORKSPACE_DIRS = ("logs", "datasets", "models")


def scaffold_workspace(target: Path, force: bool = False) -> dict:
    """Create the IMPSY workspace layout under `target`.

    Returns a dict with keys:
      - created_dirs: list[Path] of directories that were created
      - existing_dirs: list[Path] of directories that already existed
      - config_file: Path to config.toml in the target
      - config_action: 'created' | 'overwritten' | 'kept' | 'no-template'
    """
    target = target.expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)

    created_dirs = []
    existing_dirs = []
    for name in WORKSPACE_DIRS:
        d = target / name
        if d.exists():
            existing_dirs.append(d)
        else:
            d.mkdir(parents=True)
            created_dirs.append(d)

    config_file = target / "config.toml"
    template = default_config_template()
    if template is None:
        config_action = "no-template"
    elif config_file.exists() and not force:
        config_action = "kept"
    else:
        action = "overwritten" if config_file.exists() else "created"
        config_file.write_text(template, encoding="utf-8")
        config_action = action

    return {
        "target": target,
        "created_dirs": created_dirs,
        "existing_dirs": existing_dirs,
        "config_file": config_file,
        "config_action": config_action,
    }


@click.command(name="init")
@click.argument(
    "directory",
    required=False,
    type=click.Path(file_okay=False, path_type=Path),
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Overwrite an existing config.toml. Existing logs/, datasets/, models/ "
    "directories are always left in place regardless of this flag.",
)
def init(directory, force):
    """Scaffold an IMPSY workspace (logs/, datasets/, models/, config.toml).

    Without a positional argument, initialises the current directory. The
    bundled default config template is copied to config.toml unless one
    already exists (use --force to overwrite).
    """
    target = directory if directory is not None else Path.cwd()
    result = scaffold_workspace(target, force=force)

    click.secho(f"Workspace: {result['target']}", fg="blue")
    for d in result["created_dirs"]:
        click.secho(f"  created   {d.name}/", fg="green")
    for d in result["existing_dirs"]:
        click.secho(f"  exists    {d.name}/", fg="yellow")

    action = result["config_action"]
    cfg = result["config_file"]
    if action == "created":
        click.secho(f"  created   {cfg.name}", fg="green")
    elif action == "overwritten":
        click.secho(f"  overwrote {cfg.name}", fg="green")
    elif action == "kept":
        click.secho(
            f"  exists    {cfg.name} (use --force to overwrite)", fg="yellow"
        )
    else:  # no-template
        click.secho(
            "  warning   bundled default template not found; "
            "config.toml not written",
            fg="red",
        )

    click.secho("", fg="blue")
    click.secho("Next steps:", fg="blue")
    click.secho(f"  1. Edit {cfg} to match your I/O setup", fg="blue")
    click.secho("  2. `impsy run` to start logging interactions", fg="blue")
    click.secho(
        "  3. `impsy dataset` then `impsy train` once you have logs to train on",
        fg="blue",
    )
