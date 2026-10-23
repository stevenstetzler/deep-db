import click
from deep_db.cli import main

@main.group()
@click.pass_context
def fetch(ctx):
    pass

@fetch.command()
@click.argument("mpc_db", type=str, help="SQLAlchemy URL to MPC SBN database")
@click.option("--night", multiple=True, type=int, help="")
@click.option("--field", multiple=True, type=str, help="")
@click.option("--expnum", multiple=True, type=int, help="")
@click.option("--output", type=str, default=None, help="")
@click.option("--format", type=str, default='ascii.csv', help="")
@click.pass_context
def mpc(ctx, mpc_db, night, field, expnum, output, format):
    from deep_db.fetch.mpc import fetch_mpc
    from deep_db.query import query_survey
    from deep_db.database import get_db
    from deep_db.database.survey import Exposure
    from sqlalchemy.orm import Session
    db = get_db()
    with Session(db) as session:
        exposure_ids = list(query_survey(session, ["Exposure.id"], night=night, field=field, expnum=expnum))
        exposure_ids = list(map(lambda x : x[0], exposure_ids))
        exposures = session.query(Exposure).filter(Exposure.id.in_(exposure_ids))
        fetch_mpc(mpc_db, exposures, output=output, format=format)

@fetch.command()
@click.argument("mpc_db", type=str, help="SQLAlchemy URL to MPC SBN database")
@click.option("--night", multiple=True, type=int, help="")
@click.option("--field", multiple=True, type=str, help="")
@click.option("--expnum", multiple=True, type=int, help="")
@click.option("--output", type=str, default=None, help="")
@click.option("--format", type=str, default='ascii.csv', help="")
@click.pass_context
def mpchecker(ctx, mpc_db, night, field, expnum, output, format):
    from deep_db.fetch.mpchecker import fetch_mpchecker
    from deep_db.query import query_survey
    from deep_db.database import get_db
    from deep_db.database.survey import Exposure
    from sqlalchemy.orm import Session
    db = get_db()
    with Session(db) as session:
        exposure_ids = list(query_survey(session, ["Exposure.id"], night=night, field=field, expnum=expnum))
        exposure_ids = list(map(lambda x : x[0], exposure_ids))
        exposures = session.query(Exposure).filter(Exposure.id.in_(exposure_ids))
        fetch_mpchecker(mpc_db, exposures, output=output, format=format)
