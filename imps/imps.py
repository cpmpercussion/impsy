"""imps.imps: provides entry point main() to imps.""" 


import click
from .dataset import dataset
from .train import train
from .interaction import run
from .tests import test_mdrnn

@click.group()
def cli():
    pass


def main():
    """The entry point function for IMPS, this just passes through the interfaces for each command"""
    cli.add_command(dataset)
    cli.add_command(train)
    cli.add_command(run)
    cli.add_command(test_mdrnn)
    # runs the command line interface
    cli()