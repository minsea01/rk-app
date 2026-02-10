#!/usr/bin/env bash
# Deprecated compatibility wrapper.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_ROOT/scripts/lib/deprecation.sh"

warn_deprecated "cloud_training/train_runner.sh" "scripts/train.sh"
exec "$REPO_ROOT/scripts/train.sh" "$@"

