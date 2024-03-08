"""imps.imps: provides entry point main() to imps.""" 


__version__ = "0.5.0"


import click
from .dataset import dataset
from .train import train
from .interaction import run


@click.group()
def cli():
    pass


def main():
    """The entry point function for IMPS, this just passes through the interfaces for each command"""
    cli.add_command(dataset)
    cli.add_command(train)
    cli.add_command(run)
    # runs the command line interface
    cli()