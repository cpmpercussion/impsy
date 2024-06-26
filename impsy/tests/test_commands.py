from click.testing import CliRunner
from impsy.impsy import cli


def test_main_command():
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 0


def test_model_test_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["test-mdrnn"])
    # assert result.exit_code == 0
