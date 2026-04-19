#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "Launching mpdb torch demo with accelerate."
echo "If you only want a local multiprocessing example, run:"
echo "  python \"$script_dir/demo_mpdb.py\" --mode file"

(echo "value = 'updated-from-launcher'"; echo "c"; echo "c"; echo "c") | accelerate launch "$script_dir/demo_mpdb.py" --mode torch
