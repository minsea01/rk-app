#!/usr/bin/env bash

# Standard deprecation helper for shell wrappers.
warn_deprecated() {
  local old="$1"
  local new="$2"
  local removal="${3:-2.0.0}"
  echo "[DEPRECATED] ${old} -> ${new}, removal in ${removal}" >&2
}

