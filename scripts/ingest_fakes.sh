#!/usr/bin/env bash
# Ingest the DEEP fakes-injection pipeline's FITS tables (asteroidorbits.fits,
# fakeasteroids.fits, fakes_orbits.fits, fakes_detections.fits, and
# optionally fakes_binaryproperties.fits) into synthetic_object_designation,
# synthetic_object, synthetic_object_binary, and synthetic_object_ephemeris.
#
# Requires `deep-db ingest survey` to have already been run first --
# detections whose (EXPNUM, CCDNUM) doesn't resolve to an ingested
# DetectorExposure are skipped (with a warning); re-run this script after
# backfilling survey data to pick those up.
#
# Usage: DATA_DIR=data/fakes/DEEP scripts/ingest_fakes.sh
set -euo pipefail

DATA_DIR="${DATA_DIR:-data/fakes/DEEP}"

deep-db ingest fakes "$DATA_DIR"
