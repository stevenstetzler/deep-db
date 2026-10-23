#!/usr/bin/env bash
# Query the MPC database over every exposure in the survey and ingest the results.
#
# Usage: J=8 MPC_USER=... MPC_PASS=... scripts/ingest_mpc.sh
set -euo pipefail

J="${J:-4}"

mkdir -p data/mpc/expnum

deep-db query survey expnum | parallel -j "$J" \
    "deep-db query --filter expnum={} mpc --mpc postgresql://$MPC_USER:$MPC_PASS@epyc/mpc_sbn > data/mpc/expnum/{}.psv"

cat data/mpc/expnum/*.psv | deep-db ingest mpc
