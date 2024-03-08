"""imps.imps: provides entry point main() to imps.""" 


__version__ = "0.5.0"


import click
from .dataset import dataset
from .train import train
# import interaction
# import train

@click.group()
def cli():
    pass


def main():
    cli.add_command(dataset)
    cli.add_command(train)
    # TODO: the interaction and train commands
    # runs the command line interface
    cli()