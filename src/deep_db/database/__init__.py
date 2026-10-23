import os
import sys
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

_engine = None

DB_NAME = "deep_database.sqlite"
REMOTE_DB_URL = f"https://epyc.astro.washington.edu/~stevengs/DEEP/database/{DB_NAME}"
def download_db():
    # download sqlite database from URL to local cache location using xdg_dir if available
    import requests
    from pathlib import Path
    import os
    xdg_cache_home = os.getenv("XDG_CACHE_HOME", Path.home() / ".cache")
    cache_dir = Path(xdg_cache_home) / "deep_db"
    cache_dir.mkdir(parents=True, exist_ok=True)
    db_path = cache_dir / DB_NAME
    response = requests.get(REMOTE_DB_URL + "-SHA", timeout=30)
    response.raise_for_status()
    remote_sha = response.text.strip()
    if db_path.exists():
        local_sha = os.popen(f"sha256sum {db_path}").read().split()[0]
        if local_sha == remote_sha:
            logger.debug("Local database is up to date, skipping download")
            return str(db_path)
    
    logger.debug("Downloading remote database from %s to %s", REMOTE_DB_URL, db_path)
    response = requests.get(REMOTE_DB_URL, timeout=30)
    response.raise_for_status()
    with open(db_path, "wb") as f:
        f.write(response.content)
    return str(db_path)

def _get_db():
    import os
    deep_db = os.environ.get("DEEP_DB", None)
    if deep_db is None:
        db_path = download_db()
        deep_db = f"sqlite:///{db_path}"
    return deep_db

def ensure_echo_to_stderr():
    """
    SQLAlchemy's echo=True attaches its own default log handler directly
    to the "sqlalchemy.engine.Engine" logger the first time it's used,
    pointed at sys.stdout (see sqlalchemy.log._add_default_handler) --
    which would interleave SQL into a piped query's actual output.
    Pre-attaching our own stderr handler (and disabling propagation, so
    the root handler from this module's logging.basicConfig above
    doesn't *also* print it) heads that off: SQLAlchemy only adds its
    default handler when the logger has none yet, so echo=True then
    lands on stderr, exactly once. Call this before any create_engine(echo=True).
    """
    logger = logging.getLogger("sqlalchemy.engine.Engine")
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(logging.INFO)

def get_db(echo=False, new=False):
    from sqlalchemy import create_engine
    from deep_db.database.models import Base
    from deep_db.database import mpc, mpchecker, survey, synthetic_objects
    global _engine
    if echo:
        ensure_echo_to_stderr()
    if _engine is None:
        _engine = create_engine(_get_db(), echo=echo)
        Base.metadata.create_all(_engine)
    if not new:
        engine = _engine
    else:
        engine = create_engine(_get_db(), echo=echo)
    return engine
