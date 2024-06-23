import click
import time
from .utils import mdrnn_config
import pandas as pd

pd.set_option("display.float_format", lambda x: "%.4f" % x)


def build_network(dimension=4, units=64, mixes=5, layers=2):
    """Build an MDRNN model."""
    from . import mdrnn

    mdrnn.MODEL_DIR = "./models/"
    net = mdrnn.PredictiveMusicMDRNN(
        mode=mdrnn.NET_MODE_RUN,
        dimension=dimension,
        n_hidden_units=units,
        n_mixtures=mixes,
        layers=layers,
    )
    return net


@click.command(name="test-mdrnn")
def test_mdrnn():
    """This command simply loads the MDRNN to test that it works and how long it takes."""
    click.secho("Building MDRNN.")
    start_build = time.time()
    model_config = mdrnn_config("s")
    build_network(
        4,
        model_config["units"],
        model_config["mixes"],
        model_config["layers"],
    )
    click.secho(f"Done in {round(time.time() - start_build, 2)}s.")


@click.command(name="test-speed")
def prediction_speed_test():
    """This command runs a speed test experiment with different sized MDRNN models. The output is written to a CSV file."""
    from . import mdrnn

    def request_rnn_prediction(input_value, net):
        """Accesses a single prediction from the RNN."""
        start = time.time()
        output_value = net.generate_touch(input_value)
        time_delta = time.time() - start
        return output_value, time_delta

    def run_test(tests, config):
        times = []
        net = build_network(
            config["dimension"],
            config["units"],
            config["mixes"],
            config["layers"],
        )
        for i in range(tests):
            ## Predictions.
            item = mdrnn.random_sample(out_dim=config["dimension"])
            rnn_output, t = request_rnn_prediction(item, net)
            out_dict = {
                "time": t,
                "mixes": config["mixes"],
                "layers": config["layers"],
                "units": config["units"],
                "dimension": config["dimension"],
            }
            times.append(out_dict)
        return times

    experiments = []
    mdrnn_units = [64, 128, 256, 512]
    dimensions = [2, 3, 4, 5, 6, 7, 8, 9]
    for un in mdrnn_units:
        for dim in dimensions:
            net_config = {"mixes": 5, "layers": 2, "units": un, "dimension": dim}
            times = run_test(100, net_config)
            experiments.extend(times)
    total_experiment = pd.DataFrame.from_records(experiments)
    total_experiment.to_csv("total_exp.csv")
    click.secho(total_experiment.describe())
