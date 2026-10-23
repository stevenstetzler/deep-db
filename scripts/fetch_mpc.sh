#!/usr/bin/env bash
# Query the MPC database over every exposure in the survey and ingest the results.
#
# Usage: J=8 MPC_USER=... MPC_PASS=... scripts/ingest_mpc.sh

set -euo pipefail

J="${J:-4}"
export MPC_USER MPC_PASS

mkdir -p data/fetch_mpc/expnum

fetch_one() {
    local expnum="$1"
    local out="data/fetch_mpc/expnum/${expnum}.ecsv"

    # Skip if the file exists and is non-empty
    if [[ -s "$out" ]]; then
        return 0
    fi

    # Write to a temp file, then move into place only on success
    local tmp="${out}.tmp.$$"
    if deep-db fetch mpc "postgresql://$MPC_USER:$MPC_PASS@epyc/mpc_sbn" \
            --expnum "$expnum" --format 'ascii.ecsv' > "$tmp"; then
        mv "$tmp" "$out"
    else
        rm -f "$tmp"
        return 1
    fi
}
export -f fetch_one

deep-db query survey expnum | parallel -j "$J" fetch_one {}
