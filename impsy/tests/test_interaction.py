from impsy import interaction
from impsy import utils
import pytest
from pathlib import Path

@pytest.fixture(scope="session")
def default_config():
    """get the default config file."""
    config_path = Path("configs") / "default.toml"
    config = utils.get_config_data(config_path)
    return(config)


def test_interaction_server(default_config):
    """Just tests creation of an interaction server object"""
    interaction_server = interaction.InteractionServer(default_config)


def test_logging():
    """Just sets up logging"""
    interaction.setup_logging(2)
