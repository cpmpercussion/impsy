import click
import time
from .utils import mdrnn_config
import pandas as pd

pd.set_option("display.float_format", lambda x: "%.4f" % x)


def time_network_build(dimension, size):
    click.secho("Test: timing an MDRNN build...")
    model_config = mdrnn_config(size)
    start_build = time.time()
    from . import mdrnn
    mdrnn.build_mdrnn_model(
        dimension=dimension, 
        n_hidden_units=model_config["units"], 
        n_mixtures=model_config["mixes"], 
        n_layers=model_config["layers"],
        inference=True,
    )
    click.secho(f"Done in {round(time.time() - start_build, 2)}s.")


@click.command(name="test-mdrnn")
def test_mdrnn():
    """This command simply loads the MDRNN to test that it works and how long it takes."""
    time_network_build(4, "s")
