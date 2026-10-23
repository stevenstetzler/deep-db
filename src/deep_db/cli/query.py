import click
from deep_db.cli import main

@main.group()
@click.pass_context
def query(ctx):
    pass

@query.command()
@click.argument("fields", nargs=-1, type=str)
@click.option("--night", multiple=True, type=str)
@click.option("--field", multiple=True, type=str)
@click.option("--expnum", multiple=True, type=int)
@click.option("--detnum", multiple=True, type=int)
@click.option("--detname", multiple=True, type=str)
@click.option("--provid", multiple=True, type=str)
@click.option("--permid", multiple=True, type=str)
@click.option("--object-name", multiple=True, type=str)
@click.option("--join", type=click.Choice(["mpc_observation", "mpchecker_ephemeris", "synthetic_object_ephemeris"]), default=None)
@click.option("--output", type=str)
@click.pass_context
def survey(ctx, fields, night, field, expnum, detnum, detname, provid, permid, object_name, join, output):
    if not (fields or night or field or expnum or detnum or detname or provid or permid or object_name):
        click.echo(ctx.get_help())
        return
    from deep_db.database import get_db
    from sqlalchemy.orm import Session
    from sys import stdout, stderr
    from deep_db.database.survey import Night, Field, Exposure, Detector, DetectorExposure
    from deep_db.query import query_survey

    _output = open(output, "w") if output else stdout
    db = get_db()
    with Session(db) as session:
        q = query_survey(
            session, fields,
            night=night, field=field, expnum=expnum, detnum=detnum, detname=detname,
            provid=provid, permid=permid, object_name=object_name, join=join,
        )
        for row in q:
            print("|".join(map(str, row)), file=_output)

    if _output is not stdout:
        _output.close()

@query.group()
@click.pass_context
def observations(ctx):
    pass

@query.command()
@click.argument("fields", nargs=-1, type=str)
@click.option("--provid", multiple=True, type=str)
@click.option("--permid", multiple=True, type=str)
@click.option("--name", multiple=True, type=str)
@click.option("--night", multiple=True, type=str)
@click.option("--field", multiple=True, type=str)
@click.option("--expnum", multiple=True, type=int)
@click.option("--detnum", multiple=True, type=int)
@click.option("--detname", multiple=True, type=str)
@click.option("--output", type=str)
@click.pass_context
def objects(ctx, fields, provid, permid, name, night, field, expnum, detnum, detname, output):
    if not (fields or provid or permid or name or night or field or expnum or detnum or detname):
        click.echo(ctx.get_help())
        return
    from deep_db.database import get_db
    from sqlalchemy.orm import Session
    from sys import stdout
    from deep_db.query import query_known_objects

    _output = open(output, "w") if output else stdout
    db = get_db()
    with Session(db) as session:
        q = query_known_objects(
            session, fields,
            provid=provid, permid=permid, name=name,
            night=night, field=field, expnum=expnum, detnum=detnum, detname=detname,
        )
        for row in q:
            print("|".join(map(str, row)), file=_output)

    if _output is not stdout:
        _output.close()

@query.command()
@click.argument("fields", nargs=-1, type=str)
@click.option("--provid", multiple=True, type=str)
@click.option("--permid", multiple=True, type=str)
@click.option("--name", multiple=True, type=str)
@click.option("--night", multiple=True, type=str)
@click.option("--field", multiple=True, type=str)
@click.option("--expnum", multiple=True, type=int)
@click.option("--detnum", multiple=True, type=int)
@click.option("--detname", multiple=True, type=str)
@click.option("--output", type=str)
@click.pass_context
def object_ephemeris(ctx, fields, provid, permid, name, night, field, expnum, detnum, detname, output):
    if not (fields or provid or permid or name or night or field or expnum or detnum or detname):
        click.echo(ctx.get_help())
        return
    from deep_db.database import get_db
    from sqlalchemy.orm import Session
    from sys import stdout
    from deep_db.query import query_known_object_ephemeris

    _output = open(output, "w") if output else stdout
    db = get_db()
    with Session(db) as session:
        q = query_known_object_ephemeris(
            session, fields,
            provid=provid, permid=permid, name=name,
            night=night, field=field, expnum=expnum, detnum=detnum, detname=detname,
        )
        for row in q:
            print("|".join(map(str, row)), file=_output)

    if _output is not stdout:
        _output.close()

@query.command()
@click.argument("fields", nargs=-1, type=str)
@click.option("--orbit-id", multiple=True, type=int)
@click.option("--name", multiple=True, type=str)
@click.option("--source", multiple=True, type=str)
@click.option("--component", multiple=True, type=str)
@click.option("--night", multiple=True, type=str)
@click.option("--field", multiple=True, type=str)
@click.option("--expnum", multiple=True, type=int)
@click.option("--detnum", multiple=True, type=int)
@click.option("--detname", multiple=True, type=str)
@click.option("--output", type=str)
@click.pass_context
def fakes(ctx, fields, orbit_id, name, source, component, night, field, expnum, detnum, detname, output):
    if not (fields or orbit_id or name or source or component or night or field or expnum or detnum or detname):
        click.echo(ctx.get_help())
        return
    from deep_db.database import get_db
    from sqlalchemy.orm import Session
    from sys import stdout
    from deep_db.query import query_synthetic_objects

    _output = open(output, "w") if output else stdout
    db = get_db()
    with Session(db) as session:
        q = query_synthetic_objects(
            session, fields,
            orbit_id=orbit_id, name=name, source=source, component=component,
            night=night, field=field, expnum=expnum, detnum=detnum, detname=detname,
        )
        for row in q:
            print("|".join(map(str, row)), file=_output)

    if _output is not stdout:
        _output.close()

@query.command()
@click.argument("fields", nargs=-1, type=str)
@click.option("--orbit-id", multiple=True, type=int)
@click.option("--name", multiple=True, type=str)
@click.option("--source", multiple=True, type=str)
@click.option("--component", multiple=True, type=str)
@click.option("--night", multiple=True, type=str)
@click.option("--field", multiple=True, type=str)
@click.option("--expnum", multiple=True, type=int)
@click.option("--detnum", multiple=True, type=int)
@click.option("--detname", multiple=True, type=str)
@click.option("--output", type=str)
@click.pass_context
def fakes_ephemeris(ctx, fields, orbit_id, name, source, component, night, field, expnum, detnum, detname, output):
    if not (fields or orbit_id or name or source or component or night or field or expnum or detnum or detname):
        click.echo(ctx.get_help())
        return
    from deep_db.database import get_db
    from sqlalchemy.orm import Session
    from sys import stdout
    from deep_db.query import query_synthetic_object_ephemeris

    _output = open(output, "w") if output else stdout
    db = get_db()
    with Session(db) as session:
        q = query_synthetic_object_ephemeris(
            session, fields,
            orbit_id=orbit_id, name=name, source=source, component=component,
            night=night, field=field, expnum=expnum, detnum=detnum, detname=detname,
        )
        for row in q:
            print("|".join(map(str, row)), file=_output)

    if _output is not stdout:
        _output.close()
