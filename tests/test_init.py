"""Tests for the `impsy init` scaffolding command."""

from click.testing import CliRunner

from impsy import init as init_module
from impsy.impsy import cli


def test_scaffold_creates_dirs_and_config(tmp_path):
    result = init_module.scaffold_workspace(tmp_path)
    assert result["target"] == tmp_path.resolve()
    assert result["config_action"] == "created"
    for name in ("logs", "datasets", "models"):
        assert (tmp_path / name).is_dir()
        assert (tmp_path / name) in result["created_dirs"]
    assert result["config_file"] == tmp_path / "config.toml"
    assert (tmp_path / "config.toml").exists()
    content = (tmp_path / "config.toml").read_text()
    assert "[interaction]" in content
    assert "[model]" in content


def test_scaffold_keeps_existing_config(tmp_path):
    """Second invocation without --force must leave config.toml alone."""
    (tmp_path / "config.toml").write_text('title = "user edits"\n')
    result = init_module.scaffold_workspace(tmp_path)
    assert result["config_action"] == "kept"
    assert (tmp_path / "config.toml").read_text() == 'title = "user edits"\n'


def test_scaffold_force_overwrites_config(tmp_path):
    (tmp_path / "config.toml").write_text('title = "stale"\n')
    result = init_module.scaffold_workspace(tmp_path, force=True)
    assert result["config_action"] == "overwritten"
    new_content = (tmp_path / "config.toml").read_text()
    assert "[interaction]" in new_content
    assert "stale" not in new_content


def test_scaffold_idempotent_directories(tmp_path):
    """Pre-existing logs/ etc. are reported as existing, not re-created."""
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "old.log").write_text("keep me\n")
    result = init_module.scaffold_workspace(tmp_path)
    assert (tmp_path / "logs") in result["existing_dirs"]
    assert (tmp_path / "logs") not in result["created_dirs"]
    # Existing contents must be preserved.
    assert (tmp_path / "logs" / "old.log").read_text() == "keep me\n"


def test_init_cli_default_target_is_cwd(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runner = CliRunner()
    result = runner.invoke(cli, ["init"])
    assert result.exit_code == 0, result.output
    assert (tmp_path / "logs").is_dir()
    assert (tmp_path / "config.toml").exists()
    assert "Next steps" in result.output


def test_init_cli_with_positional_directory(tmp_path):
    """Positional DIRECTORY arg targets that path, not CWD."""
    target = tmp_path / "my-workspace"
    runner = CliRunner()
    result = runner.invoke(cli, ["init", str(target)])
    assert result.exit_code == 0, result.output
    assert target.is_dir()
    assert (target / "logs").is_dir()
    assert (target / "config.toml").exists()


def test_init_cli_refuses_to_clobber_without_force(tmp_path):
    (tmp_path / "config.toml").write_text('title = "mine"\n')
    runner = CliRunner()
    result = runner.invoke(cli, ["init", str(tmp_path)])
    assert result.exit_code == 0
    assert "use --force" in result.output
    assert (tmp_path / "config.toml").read_text() == 'title = "mine"\n'


def test_init_cli_force_overwrites(tmp_path):
    (tmp_path / "config.toml").write_text('title = "stale"\n')
    runner = CliRunner()
    result = runner.invoke(cli, ["init", str(tmp_path), "--force"])
    assert result.exit_code == 0
    content = (tmp_path / "config.toml").read_text()
    assert "[interaction]" in content
    assert "stale" not in content


def test_init_cli_registered_in_group():
    """`impsy init --help` must work via the top-level cli group."""
    runner = CliRunner()
    result = runner.invoke(cli, ["init", "--help"])
    assert result.exit_code == 0
    assert "Scaffold an IMPSY workspace" in result.output
    assert "--force" in result.output
