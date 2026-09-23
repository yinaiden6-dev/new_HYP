#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
export TEXMFVAR="$PWD/.texlive-var"
export TEXMFCONFIG="$PWD/.texlive-config"
if ! kpsewhich xelatex.fmt >/dev/null; then
    fmtutil-user --byfmt xelatex
fi
for briefing_language in zh en; do
    for briefing_pass in 1 2; do
        xelatex -interaction=nonstopmode -halt-on-error -no-shell-escape "main_${briefing_language}.tex"
    done
done
