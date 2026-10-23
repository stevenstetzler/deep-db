import click
import logging
from deep_db.cli import main

logging.basicConfig()
logger = logging.getLogger(__name__)

@main.group()
@click.pass_context
def database(ctx):
    pass

@database.command()
@click.argument("url", type=str, help="SQLAlchemy database url")
@click.pass_context
def create(ctx, url):
    """
    Create the database given a sqlalchemy string
    """
    from sqlalchemy import create_engine
    from deep_db.database.models import Base
    from deep_db.database import mpc, mpchecker, survey, synthetic_objects
    Base.metadata.create_all(create_engine(url))

@database.command()
@click.argument("url", type=str, help="SQLAlchemy database url")
@click.option("--batch-size", type=int, default=10_000)
@click.pass_context
def export(ctx, url, batch_size, echo=False):
    """
    Copy every table in the local DEEP_DB database to `target_url`,
    translated through SQLAlchemy Core so the target can be any dialect
    SQLAlchemy supports (sqlite:///, postgresql://, ...) -- not just a
    byte-for-byte copy of the source file.

    Tables are created (via models.Base.metadata.create_all) and
    populated in `Base.metadata.sorted_tables` order, which SQLAlchemy
    already topologically sorts by foreign key, so a child table's rows
    are never inserted before the parent row they reference.

    Each source table is reflected from the live database rather than
    assumed to match models.py column-for-column: a table created by an
    older model definition (most plausible for one of the tables nothing
    in ingest.py/query.py ever writes to -- see CODE.md's "Unused ORM
    models") can be missing a column the current model has added. Columns
    missing from the source are left NULL in the target; columns in the
    source but not in the current model are dropped. A table the current
    model defines but the source database doesn't have at all is skipped.
    """
    from sqlalchemy import create_engine, select, inspect, MetaData, Table
    from sqlalchemy.engine import make_url
    from deep_db.database.models import Base
    from deep_db.database import get_db, ensure_echo_to_stderr

    if echo:
        ensure_echo_to_stderr()

    # see query.py:query_mpc for why this is needed against postgres --
    # not a recognized create_engine() argument for other dialects.
    target_engine_kwargs = {}
    if make_url(url).get_backend_name() == 'postgresql':
        target_engine_kwargs['client_encoding'] = 'utf8'

    source_engine = get_db(echo=echo)
    target_engine = create_engine(url, echo=echo, **target_engine_kwargs)
    Base.metadata.create_all(target_engine)

    source_table_names = set(inspect(source_engine).get_table_names())
    source_metadata = MetaData()

    with source_engine.connect() as source_conn:
        for table in Base.metadata.sorted_tables:
            if table.name not in source_table_names:
                logger.warning("Source database has no table %r; skipping", table.name)
                continue

            source_table = Table(table.name, source_metadata, autoload_with=source_engine)
            columns = [c.name for c in source_table.columns if c.name in table.columns]
            missing = [c.name for c in table.columns if c.name not in source_table.columns]
            if missing:
                logger.warning(
                    "Source table %r is missing column(s) %s present in the "
                    "target schema; leaving them NULL", table.name, missing,
                )

            result = source_conn.execute(select(*[source_table.c[name] for name in columns]))
            total = 0
            with target_engine.begin() as target_conn:
                while True:
                    batch = result.fetchmany(batch_size)
                    if not batch:
                        break
                    target_conn.execute(table.insert(), [dict(zip(columns, row)) for row in batch])
                    total += len(batch)
            if total:
                logger.info("Exported %d rows from %s", total, table.name)
