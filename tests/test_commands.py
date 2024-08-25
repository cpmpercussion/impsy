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

def test_tflite_converter_command(models_location, keras_file, weights_file):
    runner = CliRunner()
    result = runner.invoke(cli, ["convert-tflite", "--out_dir", str(models_location)])
    result = runner.invoke(cli, ["convert-tflite", "-model", str(keras_file),"--out_dir", str(models_location)])
    result = runner.invoke(cli, ["convert-tflite", "-model", str(weights_file),"--out_dir", str(models_location)])

# def test_train_command():
    # runner = CliRunner()
    # result = runner.invoke(cli, ["train", "--out_dir", str(models_location)])

# def test_dataset_command():
#     runner = CliRunner()
#     result = runner.invoke(cli, ["dataset"])

def test_run_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["run"])

def test_webui_command():
    runner = CliRunner()
    result = runner.invoke(cli, ["webui"])
