#!/usr/bin/env bash
# Serves web/ on http://127.0.0.1:8765; open index.html?mode=... (see AGENTS.md).
set -euo pipefail
cd "$(dirname "$0")/../web"
exec python3 -I -m http.server "${PORT:-8765}" --bind 127.0.0.1
