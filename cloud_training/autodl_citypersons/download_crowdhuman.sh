#!/usr/bin/env bash
# Deprecated wrapper for cloud_training/download_crowdhuman.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
source "$REPO_ROOT/scripts/lib/deprecation.sh"

warn_deprecated \
  "cloud_training/autodl_citypersons/download_crowdhuman.sh" \
  "cloud_training/download_crowdhuman.sh"

exec "$REPO_ROOT/cloud_training/download_crowdhuman.sh" "$@"

