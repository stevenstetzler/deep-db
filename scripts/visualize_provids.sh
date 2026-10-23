#!/usr/bin/env bash
# Generate known-object plots for every object in the deep database.
# Skips objects whose plot output already exists (set FORCE=1 to regenerate everything).
# Usage: J=16 ./visualize.sh   (J = number of parallel jobs, default 8)
set -euo pipefail

# Project root is two levels up from this script (e.g. <root>/scripts/visualize.sh)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

export DEEP_DB="sqlite:///deep_database.sqlite"
cd "$PROJECT_DIR"

# `mamba activate` doesn't work in non-interactive shells until the shell hook is loaded.
# (set +u: activation scripts often reference unset variables)
set +u
# eval "$(mamba shell hook --shell bash)"
# mamba activate bash
source env/bin/activate
set -u

PLOT_DIR="plots"
mkdir -p "${PLOT_DIR}"

# Plot a single object, unless output for it already exists.
# Existing output = plots/NAME (file or directory) or plots/NAME.<ext>
plot_one() {
  local provid="$1"
  local out="${PLOT_DIR}/${provid// /_}"
  if [[ -z "${FORCE:-}" ]] && { [[ -e "$out" ]] || compgen -G "${out}.*" > /dev/null; }; then
    echo "skip   $provid"
    return 0
  fi
  echo "plot   $provid"
  deep-db visualize known-objects --provid "$provid" --output "$out" --plot-ccds
}
export -f plot_one

# One provid per line; parallel splits on newlines, so provids containing spaces are safe.
/nvme/users/stevengs/mamba/bin/sqlite3 -list -noheader deep_database.sqlite 'select distinct provid from mpc_object join mpc_observation on mpc_object.id = mpc_observation.object_id' \
  | parallel -j "${J:-8}" plot_one {}