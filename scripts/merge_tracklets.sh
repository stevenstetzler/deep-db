#!/usr/bin/env bash
# Run deep-db merge-tracklets over every night in the survey, in parallel.
#
# Usage: J=8 scripts/merge_tracklets.sh
set -euo pipefail

J="${J:-4}"

deep-db query survey night | parallel -j "$J" \
    'deep-db merge-tracklets --night {}'
