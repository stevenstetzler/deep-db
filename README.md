# Database of Moving Objects for the DECam Ecliptic Exploration Project

`deep-db` is a Python/SQLAlchemy project for building and querying a moving-object database for DECam observations in the DECam Ecliptic Exploration Project (DEEP).

It provides workflows to ingest observational metadata, object/orbit/ephemeris data, and repository-based detector locations, then generate downstream products like cutouts and photometry.

## Documentation
- Repository summary: `docs/summary.md`
- Database schema: `docs/schema.md`

## Query examples
Run queries using the query CLI module:

```bash
# Query exposures from a local SQLite database
DEEP_DB="sqlite:///deep_db.db" python -m deep_db.query exposures

# Query exposures for a specific field and night
DEEP_DB="sqlite:///deep_db.db" python -m deep_db.query exposures \
  --filter Field.name=A0a night=20190401

# Query objects with selected output columns
DEEP_DB="sqlite:///deep_db.db" python -m deep_db.query objects \
  --filter type=mpc \
  --query SolarSystemObject.name Exposure.expnum Ephemeris.ra Ephemeris.dec
```
