#!/usr/bin/env bash
#
# Local mirror of the "Build and Deploy Jekyll (al-folio)" GitHub Action.
#
# This reproduces the *build* half of the workflow on your machine (it does not
# publish anything). It installs gems and runs the same production Jekyll build
# the action runs, writing the static site to ./_site.
#
# Usage:
#   scripts/build_site.sh          # build into _site
#   scripts/build_site.sh serve    # build and serve at http://localhost:4000
#
# Ruby/Bundler are provided here via a conda env named "jekyll". If you manage
# Ruby another way, just ensure `bundle` is on PATH and skip the conda block.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# --- Ruby toolchain (conda) -------------------------------------------------
if ! command -v bundle >/dev/null 2>&1; then
  if command -v conda >/dev/null 2>&1; then
    # shellcheck disable=SC1091
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate jekyll
  fi
fi

if ! command -v bundle >/dev/null 2>&1; then
  echo "error: 'bundle' not found. Create the env first, e.g.:" >&2
  echo "  conda create -y -n jekyll -c conda-forge ruby'>=3.2' compilers make" >&2
  echo "  conda activate jekyll && gem install bundler jekyll" >&2
  exit 1
fi

# --- Mirror the CI steps ----------------------------------------------------
# (CI uses `bundle config set deployment true`; locally we keep it relaxed so
#  the Gemfile.lock can be generated/updated on first run.)
bundle install --jobs 4

if [[ "${1:-build}" == "serve" ]]; then
  JEKYLL_ENV=production bundle exec jekyll serve --trace
else
  JEKYLL_ENV=production bundle exec jekyll build --trace
  echo
  echo "Built site -> $REPO_ROOT/_site"
fi
