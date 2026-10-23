import click
from deep_db.cli import main
from deep_db.cli.util import MultiValueOption

@main.group
@click.pass_context
def visualize(ctx):
    pass

@visualize.command()
@click.option("--provid", cls=MultiValueOption, multiple=True, type=str, help="Unpacked MPC provisional designation")
@click.option("--permid", cls=MultiValueOption, multiple=True, type=str, help="MPC permanent designation")
@click.option("--name", cls=MultiValueOption, multiple=True, type=str, help="MPC object name")
@click.option("--night", cls=MultiValueOption, multiple=True, type=int, help="Survey night (e.g. 20190402)")
@click.option("--field", cls=MultiValueOption, multiple=True, type=str, help="Survey field name (e.g. A0a)")
@click.option("--expnum", cls=MultiValueOption, multiple=True, type=int, help="Exposure number")
@click.option("--detnum", cls=MultiValueOption, multiple=True, type=int, help="Detector number")
@click.option("--detname", cls=MultiValueOption, multiple=True, type=str, help="Detector name")
@click.option("--plot-ccds", is_flag=True, help="Outline the CCDs the observations landed on, labeled by (night, field)")
@click.option("--annotate-ccds", is_flag=True, help="Label each CCD box with its detector (implies --plot-ccds)")
@click.option("--output", type=str, help="Path to save the figure to (shows interactively if omitted)")
@click.pass_context
def known_objects(ctx, provid, permid, name, night, field, expnum, detnum, detname, plot_ccds, annotate_ccds, output):
    from sqlalchemy.orm import Session
    from deep_db.database import get_db
    from deep_db.visualize import visualize_known_objects

    engine = get_db()
    with Session(engine) as session:
        visualize_known_objects(
            session,
            provid=provid or None,
            permid=permid or None,
            name=name or None,
            night=night or None,
            field=field or None,
            expnum=expnum or None,
            detnum=detnum or None,
            detname=detname or None,
            plot_ccds=plot_ccds or annotate_ccds,
            annotate_ccds=annotate_ccds,
            output=output,
        )

@visualize.command()
@click.option("--orbit-id", cls=MultiValueOption, multiple=True, type=int, help="SyntheticObjectDesignation orbit_id")
@click.option("--name", cls=MultiValueOption, multiple=True, type=str, help="SyntheticObjectDesignation name")
@click.option("--night", cls=MultiValueOption, multiple=True, type=int, help="Survey night (e.g. 20190402)")
@click.option("--field", cls=MultiValueOption, multiple=True, type=str, help="Survey field name (e.g. A0a)")
@click.option("--expnum", cls=MultiValueOption, multiple=True, type=int, help="Exposure number")
@click.option("--detnum", cls=MultiValueOption, multiple=True, type=int, help="Detector number")
@click.option("--detname", cls=MultiValueOption, multiple=True, type=str, help="Detector name")
@click.option("--plot-ccds", is_flag=True, help="Outline the CCDs the ephemerides landed on, labeled by (night, field)")
@click.option("--annotate-ccds", is_flag=True, help="Label each CCD box with its detector (implies --plot-ccds)")
@click.option("--output", type=str, help="Path to save the figure to (shows interactively if omitted)")
@click.pass_context
def synthetic_objects(ctx, orbit_id, name, night, field, expnum, detnum, detname, plot_ccds, annotate_ccds, output):
    from sqlalchemy.orm import Session
    from deep_db.database import get_db
    from deep_db.visualize import visualize_synthetic_objects

    engine = get_db()
    with Session(engine) as session:
        visualize_synthetic_objects(
            session,
            orbit_id=orbit_id or None,
            name=name or None,
            night=night or None,
            field=field or None,
            expnum=expnum or None,
            detnum=detnum or None,
            detname=detname or None,
            plot_ccds=plot_ccds or annotate_ccds,
            annotate_ccds=annotate_ccds,
            output=output,
        )
