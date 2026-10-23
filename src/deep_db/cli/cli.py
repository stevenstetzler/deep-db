import click

@click.group()
@click.option("--log-level", default="INFO", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
@click.pass_context
def main(ctx, log_level):
    pass
