#!/usr/bin/env bash
set -euo pipefail

# Resolve from the script location, regardless of the invoking directory.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
if [[ -e URPC2020_YOLO ]]; then
    echo "URPC2020_YOLO already exists; refusing to overwrite it." >&2
    exit 1
fi
if [[ ! -d URPC2020 ]]; then
    unzip -n ./URPC2020.zip -d ./
fi
for source_split in train valid test; do
    for kind in images labels; do
        test -d "URPC2020/$source_split/$kind" || {
            echo "Missing URPC2020/$source_split/$kind" >&2
            exit 1
        }
    done
done
for pair in train:train valid:val test:test; do
    source_split="${pair%%:*}"
    target_split="${pair##*:}"
    for kind in images labels; do
        mkdir -p "URPC2020_YOLO/$kind/$target_split"
        cp -a "URPC2020/$source_split/$kind/." "URPC2020_YOLO/$kind/$target_split/"
    done
done
