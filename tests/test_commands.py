from click.testing import CliRunner
from impsy.impsy import cli
from impsy import interaction


def test_main_command():
    runner = CliRunner()
    result = runner.invoke(cli)
    assert result.exit_code == 2


def test_cli_registers_subcommands():
    """Importing cli should expose all subcommands without calling main()."""
    expected = {"dataset", "train", "run", "test-mdrnn", "convert-tflite", "webui"}
    assert expected.issubset(set(cli.commands.keys()))


def test_model_test_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["test-mdrnn"])
    # assert result.exit_code == 0


def test_tflite_converter_command(models_location, keras_file, weights_file):
    runner = CliRunner()
    result = runner.invoke(cli, ["convert-tflite", "--out_dir", str(models_location)])
    result = runner.invoke(
        cli,
        [
            "convert-tflite",
            "-model",
            str(keras_file),
            "--out_dir",
            str(models_location),
        ],
    )
    result = runner.invoke(
        cli,
        [
            "convert-tflite",
            "-model",
            str(weights_file),
            "--out_dir",
            str(models_location),
        ],
    )


# def test_train_command():
# runner = CliRunner()
# result = runner.invoke(cli, ["train", "--out_dir", str(models_location)])

# def test_dataset_command():
#     runner = CliRunner()
#     result = runner.invoke(cli, ["dataset"])


def test_run_command_help():
    """`impsy run --help` should list the new CLI override options."""
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "--help"])
    assert result.exit_code == 0
    for flag in (
        "--mode",
        "--threshold",
        "--input-thru",
        "--sigma-temp",
        "--pi-temp",
        "--timescale",
        "--dimension",
        "MODEL_FILE",
    ):
        assert flag in result.output, f"missing {flag} in --help output"


def test_run_command_missing_config():
    """`impsy run` with a missing config file should abort cleanly, not crash."""
    runner = CliRunner()
    with runner.isolated_filesystem():
        result = runner.invoke(cli, ["run", "-c", "does-not-exist.toml"])
    assert result.exit_code != 0
    assert "does-not-exist.toml" in result.output


def test_run_command_invokes_server(monkeypatch, tmp_path):
    """`impsy run <config>` should construct InteractionServer and call serve_forever."""
    config_file = tmp_path / "config.toml"
    config_file.write_text(
        "verbose = false\n"
        "log_input = true\n"
        "log_predictions = false\n"
        "[interaction]\n"
        'mode = "useronly"\n'
        "threshold = 0.1\n"
        "input_thru = false\n"
        "[model]\n"
        "dimension = 4\n"
        'size = "xs"\n'
        'file = "missing.tflite"\n'
        "pitemp = 1.0\n"
        "sigmatemp = 0.01\n"
        "timescale = 1\n"
    )

    served = {"count": 0}

    class FakeServer:
        def __init__(self, config, log_location):
            served["config"] = config
            served["log_location"] = log_location

        def serve_forever(self):
            served["count"] += 1

    monkeypatch.setattr(interaction, "InteractionServer", FakeServer)
    runner = CliRunner()
    result = runner.invoke(cli, ["run", "-c", str(config_file)])
    assert result.exit_code == 0, result.output
    assert served["count"] == 1
    assert served["config"]["interaction"]["mode"] == "useronly"


def test_webui_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["webui"])
