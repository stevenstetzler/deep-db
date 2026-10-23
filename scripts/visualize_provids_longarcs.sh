#!/usr/bin/env bash
# Generate known-object plots for the objects with the longest observation arcs.
# Only objects with more than 2 observations are considered, ranked by (last - first) obstime.
# Skips objects whose plot output already exists (set FORCE=1 to regenerate everything).
# Usage: N=50 J=16 ./visualize_provid_longarcs.sh
#   N = how many of the longest-arc objects to plot (default 10)
#   J = number of parallel jobs (default 8)
set -euo pipefail

# Project root is two levels up from this script (e.g. <root>/scripts/visualize_provid_longarcs.sh)
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

PLOT_DIR="plots/long_arcs"
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

read -r -d '' QUERY <<SQL || true
with cte as (
  select object_id,
         julianday(max(obstime)) - julianday(min(obstime)) as dt_days
  from mpc_observation
  where object_id is not null
  group by object_id
  having count(obstime) > 2
)
select provid
from cte
join mpc_object on cte.object_id = mpc_object.id
order by dt_days desc
limit ${N:-10};
SQL

# One provid per line, longest arcs first; parallel splits on newlines, so provids with spaces are safe.
/nvme/users/stevengs/mamba/bin/sqlite3 -list -noheader deep_database.sqlite "$QUERY" \
  | parallel -j "${J:-8}" plot_one {}