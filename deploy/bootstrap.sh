#!/usr/bin/env bash
# One-time setup on a fresh Ubuntu 22.04 or 24.04 server, such as an AWS Lightsail instance.
#
# On the server, from the repo folder:
#     bash deploy/bootstrap.sh
#
# The first run installs Docker and asks you to log out and back in (so your user can run
# docker without sudo). Run it again afterwards to start everything.
set -euo pipefail

if ! command -v make >/dev/null || ! command -v git >/dev/null; then
  sudo apt-get update -qq
  sudo apt-get install -y -qq make git
fi

if ! command -v docker >/dev/null; then
  echo "Installing Docker..."
  curl -fsSL https://get.docker.com | sudo sh
  sudo usermod -aG docker "$USER"
  echo
  echo "Docker is installed. Log out, log back in, and run this script again."
  exit 0
fi

if [ ! -f .env ]; then
  cp .env.example .env
  sed -i "s/^POSTGRES_PASSWORD=.*/POSTGRES_PASSWORD=$(openssl rand -hex 16)/" .env
  sed -i "s/^API_KEY=.*/API_KEY=$(openssl rand -hex 16)/" .env
  echo "Created .env with a random Postgres password and API key."
fi

make setup

echo
echo "Done. Check it from this server:"
echo "    curl -s localhost:8000/health"
echo "Your API key (send it as the X-API-Key header) is in .env:"
grep '^API_KEY=' .env
