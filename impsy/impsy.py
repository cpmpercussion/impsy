"""impsy.impsy: provides entry point main() to impsy.""" 


import click
from .dataset import dataset
from .train import train
from .interaction import run
from .interaction_config import start_genai_module
from .tests import test_mdrnn

@click.group()
def cli():
    pass


def main():
    """The entry point function for IMPSY, this just passes through the interfaces for each command"""
    cli.add_command(dataset)
    cli.add_command(train)
    cli.add_command(run)
    cli.add_command(start_genai_module)
    cli.add_command(test_mdrnn)
    # runs the command line interface
    cli()