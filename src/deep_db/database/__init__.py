import os
import sys
import logging
logger = logging.getLogger(__name__)

_engine = None

DB_NAME = "deep_database.sqlite"
REMOTE_DB_URL = f"https://epyc.astro.washington.edu/~stevengs/DEEP/database/{DB_NAME}"
_db_path_cache = None


def _cache_dir():
    from pathlib import Path

    xdg_cache_home = os.getenv("XDG_CACHE_HOME") or Path.home() / ".cache"
    cache_dir = Path(xdg_cache_home) / "deep_db"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def download_db():
    """
    Return the path to the cached database, downloading it if it is missing or stale.

    The expected sha256 is fetched once per process. The checksum of the local file is
    computed only when it is downloaded and recorded in a ``<db>.sha256`` sidecar, so
    later calls compare the remote hash to the sidecar instead of re-hashing the file.
    The download is streamed to a temporary file in the cache directory, hashed on the
    fly, verified, and atomically renamed into place, so an interrupted download never
    leaves a truncated database behind. A file lock serializes concurrent processes.
    """
    global _db_path_cache
    if _db_path_cache is not None:
        return _db_path_cache

    import hashlib
    import tempfile

    import requests

    cache_dir = _cache_dir()
    db_path = cache_dir / DB_NAME
    sha_path = cache_dir / (DB_NAME + ".sha256")

    try:
        response = requests.get(REMOTE_DB_URL + "-SHA", timeout=30)
        response.raise_for_status()
        remote_sha = response.text.strip().split()[0]
    except requests.RequestException:
        if db_path.exists() and sha_path.exists():
            logger.warning("Could not check remote database checksum; using cached copy")
            _db_path_cache = str(db_path)
            return _db_path_cache
        raise

    def up_to_date():
        return db_path.exists() and sha_path.exists() and sha_path.read_text().strip() == remote_sha

    lock_file = open(cache_dir / (DB_NAME + ".lock"), "w")
    try:
        try:
            import fcntl

            fcntl.flock(lock_file, fcntl.LOCK_EX)
        except ImportError:  # pragma: no cover - non-POSIX platforms
            pass
        # Re-check: another process may have finished the download while we waited.
        if not up_to_date():
            logger.info("Downloading remote database from %s to %s", REMOTE_DB_URL, db_path)
            fd, tmp_name = tempfile.mkstemp(dir=cache_dir, prefix=DB_NAME + ".", suffix=".part")
            try:
                digest = hashlib.sha256()
                with os.fdopen(fd, "wb") as f, requests.get(REMOTE_DB_URL, stream=True, timeout=30) as r:
                    r.raise_for_status()
                    for chunk in r.iter_content(chunk_size=1 << 20):
                        f.write(chunk)
                        digest.update(chunk)
                if digest.hexdigest() != remote_sha:
                    raise RuntimeError(
                        f"Downloaded database checksum {digest.hexdigest()} does not match expected {remote_sha}"
                    )
                os.replace(tmp_name, db_path)
                sha_path.write_text(remote_sha + "\n")
            finally:
                if os.path.exists(tmp_name):
                    os.unlink(tmp_name)
        else:
            logger.debug("Local database is up to date, skipping download")
    finally:
        lock_file.close()

    _db_path_cache = str(db_path)
    return _db_path_cache


def _get_db():
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
    the application's root handler doesn't *also* print it) heads that off: SQLAlchemy only adds its
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
        # Only touch the schema of a user-supplied database: the downloaded cache is read-only data.
        if os.environ.get("DEEP_DB") is not None:
            Base.metadata.create_all(_engine)
    if not new:
        engine = _engine
    else:
        engine = create_engine(_get_db(), echo=echo)
    return engine
