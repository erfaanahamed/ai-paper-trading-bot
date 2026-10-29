#!/usr/bin/env bash
set -euo pipefail

# Installs the Hermes Agent runtime using the official Nous Research installer,
# enables its authenticated local API, and prints the dashboard settings.
# This script does not install model weights or choose/pay for a model provider.

if [[ "${EUID}" -eq 0 ]]; then
  echo "Please run as your normal user, not root. Hermes should be installed for the user running the bot."
  exit 1
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "curl is required. Install curl and run this script again."
  exit 1
fi

read -r -p "Install Hermes Agent from the official Nous Research installer? [y/N] " confirm
case "${confirm,,}" in y|yes) ;; *) echo "Cancelled."; exit 0 ;; esac

echo "Installing Hermes Agent..."
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash

mkdir -p "$HOME/.hermes"
ENV_FILE="$HOME/.hermes/.env"
if [[ -f "$ENV_FILE" ]]; then
  cp "$ENV_FILE" "$ENV_FILE.backup.$(date +%Y%m%d%H%M%S)"
fi

# Generate a strong key. Python is normally present on the target VPS.
API_KEY="$(python3 -c 'import secrets; print(secrets.token_urlsafe(36))')"
# Replace only the three API-server settings, preserving other existing settings.
python3 - "$ENV_FILE" "$API_KEY" <<'PY'
from pathlib import Path
import sys
path = Path(sys.argv[1])
key = sys.argv[2]
existing = path.read_text() if path.exists() else ""
managed = {"API_SERVER_ENABLED", "API_SERVER_KEY", "API_SERVER_HOST", "API_SERVER_PORT"}
lines = [line for line in existing.splitlines() if not any(line.strip().startswith(name + "=") for name in managed)]
lines += ["", "# AI Brain local integration", "API_SERVER_ENABLED=true", f"API_SERVER_KEY={key}", "API_SERVER_HOST=127.0.0.1", "API_SERVER_PORT=8642"]
path.write_text("\n".join(lines).strip() + "\n")
path.chmod(0o600)
PY

cat <<OUTRO

Hermes Agent runtime installed and local API settings written to:
  $ENV_FILE

IMPORTANT: complete Hermes provider/model setup next. Run:
  hermes setup

If you use Nous Portal and have access, you can instead run:
  hermes setup --portal

Then start the API gateway in a terminal or service:
  hermes gateway

AI Brain dashboard settings (keep this API key private):
  API base URL: http://127.0.0.1:8642/v1
  API key: $API_KEY
  Model name: hermes-agent

The dashboard and Hermes must run on the same machine for the 127.0.0.1 URL to work.
The key is stored with file permissions 600 in ~/.hermes/.env. Do not publish it.
Hermes analysis is advisory; it does not place trades or bypass the bot's safety filters.
OUTRO
