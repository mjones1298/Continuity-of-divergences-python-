#!/usr/bin/env bash
# Smoke test of the numerical code: runs a representative fast script from every folder in a temporary
# copy of code/ and compares the numbers with the shipped results (see smoke_test.py). About 8 minutes.
# Usage: bash code/smoke_test.sh [-k SUBSTRING] [--keep]
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$here/smoke_test.py" "$@"
