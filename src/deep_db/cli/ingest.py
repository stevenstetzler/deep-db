import click
from pathlib import Path
from deep_db.cli import main

@main.group()
@click.pass_context
def ingest(ctx):
    pass

@ingest.command()
@click.argument("input", type=str, help="The file to ingest from, default stdin")
@click.pass_context
def mpc(ctx, input):
    from deep_db.database import get_db
    from deep_db.ingest.mpc import ingest_mpc
    from astropy.table import Table
    from sqlalchemy.orm import Session
    db = get_db()
    with Session(db) as session:
        t = Table.read(input)
        if len(t) > 0:
            ingest_mpc(session, t)

@ingest.command()
@click.argument("input", type=str, help="The file to ingest from, default stdin")
@click.pass_context
def mpchecker(ctx, input):
    from deep_db.database import get_db
    from deep_db.ingest.mpchecker import ingest_mpchecker
    from astropy.table import Table
    from sqlalchemy.orm import Session
    db = get_db()
    with Session(db) as session:
        t = Table.read(input)
        if len(t) > 0:
            ingest_mpchecker(session, t)

@ingest.command()
@click.argument("data_dir", type=str)
@click.pass_context
def fakes(ctx, data_dir):
    from deep_db.database import get_db
    from deep_db.ingest.synthetic_objects import ingest_fakes
    from sqlalchemy.orm import Session
    db = get_db()
    with Session(db) as session:
        ingest_fakes(session, data_dir)

@ingest.command()
@click.argument("exposures_file", type=Path, help="Astropy table with the exposure list")
@click.argument("field_table_file", type=Path, help="Astropy table with the field definition list")
@click.pass_context
def survey(ctx, exposures_file, field_table_file):
    from deep_db.ingest.survey import ingest_survey
    from astropy.table import Table
    exposures = Table.read(exposures_file)
    field_table = Table.read(field_table_file)
    ingest_survey(exposures, field_table)

