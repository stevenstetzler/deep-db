# Repository Summary

## What this repository contains
- `src/deep_db/`: Python package for ingesting, querying, and generating products from a moving-object database.
- `data/`: example/source ECSV inputs used by ingestion workflows.
- `docs/`: high-level project documentation (including this file and schema documentation).

## Purpose
`deep-db` builds and queries a DECam-focused moving-object database for the DECam Ecliptic Exploration Project (DEEP). It supports:
- ingesting exposure metadata and detector geometry,
- ingesting ephemerides and simulated object properties,
- linking objects to detector locations in data repositories,
- generating cutouts, photometry, and release-oriented query outputs.

## Technologies used
- **Python 3.9+** (`pyproject.toml`)
- **SQLAlchemy ORM** for schema definitions and database access
- **Astropy / astropy-healpix** for astronomical coordinates and metadata handling
- **NumPy / joblib** for numerical processing and parallel execution in data ingestion pipelines
- **LSST Science Pipelines (Butler/afw/geom)** in repository-ingestion and image/cutout workflows

## Databases created and used
- **Primary DEEP database** (commonly SQLite, e.g. `deep_db.db`):
  - Created from ORM metadata (`src/deep_db/models.py`) by ingestion tools such as `src/deep_db/ingest.py`, `src/deep_db/ingest_jpl.py`, and `src/deep_db/ingest_repo_data.py`.
  - Queried by tools like `src/deep_db/query.py`.
- **Published remote SQLite database**:
  - `src/deep_db/query.py` downloads a cached copy of the published `deep_db.db` release for local querying.
- **External MPC database (input source)**:
  - `src/deep_db/query_mpc.py` connects to an existing MPC database (table reflection) and writes matched results into the DEEP database.

For table-by-table schema details, see `docs/schema.md`.
