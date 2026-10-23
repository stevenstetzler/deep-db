#!/usr/bin/env bash
# Run mpchecker over every expnum in the survey and ingest the results.
#
# Usage: J=8 scripts/ingest_mpchecker.sh
set -euo pipefail

J="${J:-4}"

export JAX_PLATFORMS='cpu'

mkdir -p data/mpchecker/expnum

deep-db query survey expnum | parallel -j "$J" \
    'deep-db query mpchecker --expnum {} > data/mpchecker/expnum/{}.psv'

cat data/mpchecker/expnum/*.psv | deep-db ingest mpchecker
