#!/bin/bash
# Copy LIMEENERGYCO PDF to static/example_contract.pdf so "Try example contract"
# uses the same file that yields 3 findings (Termination, MFN, Uncapped).
# Run from repo root: ./scripts/refresh_example_contract.sh

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
SRC="$REPO_ROOT/data/CUAD_v1/full_contract_pdf/Part_III/Distributor/LIMEENERGYCO_09_09_1999-EX-10-DISTRIBUTOR AGREEMENT.PDF"
DST="$REPO_ROOT/static/example_contract.pdf"

if [ ! -f "$SRC" ]; then
  echo "Source not found: $SRC"
  exit 1
fi

cp "$SRC" "$DST"
echo "Copied LIMEENERGYCO to static/example_contract.pdf"

