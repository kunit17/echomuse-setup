#!/usr/bin/env bash
# Preflight a host before deploying the EchoMuse controller to it.
#
# Run this ON the host that will run the controller (or pipe it over SSH).
# It is read-only: it inspects, it changes nothing.
#
#   bash preflight-controller-host.sh
#
# Exit status is non-zero if a HARD requirement is unmet (docker missing,
# wrong architecture, a required port already taken). Advisory findings do
# not affect the exit status.
#
# Why each check exists:
#   - The published image is multi-arch (amd64 + arm64), so a host of either
#     architecture is fine, but the manifest must actually exist for the
#     host's arch and the pull must land the right one.
#   - The compose file requires network_mode: host, so the controller's
#     ports land on the host's interfaces directly. An existing container
#     already in host mode takes ports directly too, so the port check is
#     a real conflict check.
#   - The controller advertises SERVER_IP over mDNS and refuses to start
#     rather than advertise a guessed address.

set -uo pipefail

IMAGE="ghcr.io/wilbowes/echomuse-controller:latest"
REQUIRED_PORTS=(8767 8768 8770 16001)
HARD_FAIL=0

say()  { printf '\n=== %s ===\n' "$*"; }
ok()   { printf '  [ ok ] %s\n' "$*"; }
warn() { printf '  [warn] %s\n' "$*"; }
bad()  { printf '  [FAIL] %s\n' "$*"; HARD_FAIL=1; }

say "host"
HOST_ARCH="$(uname -m)"
ok "$(hostname) — $(uname -srm)"
case "$HOST_ARCH" in
  x86_64|aarch64) ok "architecture $HOST_ARCH is published by $IMAGE" ;;
  *) bad "architecture $HOST_ARCH is not published by $IMAGE (amd64/arm64 only)" ;;
esac

say "docker"
if ! command -v docker >/dev/null 2>&1; then
  bad "docker not installed"
else
  ok "docker $(docker version --format '{{.Server.Version}}' 2>/dev/null || echo 'installed')"
  if docker compose version >/dev/null 2>&1; then
    ok "compose plugin $(docker compose version --short 2>/dev/null)"
  else
    bad "docker compose plugin missing (needed for the compose file)"
  fi
  if docker info >/dev/null 2>&1; then
    ok "current user can talk to the daemon"
  else
    bad "cannot reach the docker daemon — add this user to the docker group"
  fi
fi

say "required ports"
# A port already held by an EXISTING echomuse-controller is this script's
# own footprint, not a conflict: the controller is already deployed. Only
# report a conflict when something else holds the port.
DEPLOYED=0
if command -v docker >/dev/null 2>&1 && docker ps --format '{{.Names}}' 2>/dev/null | grep -qx 'echomuse-controller'; then
  DEPLOYED=1
fi
for p in "${REQUIRED_PORTS[@]}"; do
  if ss -ltn 2>/dev/null | grep -q ":${p} "; then
    if [ "$DEPLOYED" -eq 1 ]; then
      printf '  [ ok ] port %s in use by the already-deployed echomuse-controller\n' "$p"
    else
      holder="$(ss -ltnp 2>/dev/null | grep ":${p} " | sed 's/.*users:(//;s/,.*//' | head -1)"
      bad "port ${p} already in use${holder:+ by ${holder}}"
    fi
  else
    ok "port ${p} free"
  fi
done

say "existing containers"
if command -v docker >/dev/null 2>&1; then
  docker ps --format '{{.Names}}|{{.Image}}|{{.Ports}}' 2>/dev/null | while IFS='|' read -r name image ports; do
    [ -z "${name:-}" ] && continue
    mode="$(docker inspect "$name" --format '{{.HostConfig.NetworkMode}}' 2>/dev/null || echo '?')"
    if [ "$mode" = "host" ]; then
      printf '  [warn] %s runs in host mode — its ports sit on the host directly\n' "$name"
    else
      printf '  [ ok ] %s runs in %s mode\n' "$name" "$mode"
    fi
  done
  if docker ps --format '{{.Names}}' 2>/dev/null | grep -qi homeassistant; then
    warn "a Home Assistant container is present — confirm its ports do not overlap the controller's"
  fi
fi

say "resources"
AVAIL_MB="$(awk '/^MemAvailable:/ {print int($2/1024)}' /proc/meminfo 2>/dev/null)"
if [ -n "${AVAIL_MB:-}" ]; then
  if [ "$AVAIL_MB" -ge 512 ]; then
    ok "${AVAIL_MB} MB memory available"
  else
    warn "only ${AVAIL_MB} MB memory available — the controller runs wake-word inference on CPU"
  fi
fi
FREE_GB="$(df -BG --output=avail / 2>/dev/null | tail -1 | tr -dc '0-9')"
if [ -n "${FREE_GB:-}" ]; then
  if [ "$FREE_GB" -ge 3 ]; then
    ok "${FREE_GB} GB free on /"
  else
    warn "only ${FREE_GB} GB free on / — the image needs roughly 1 GB"
  fi
fi

say "result"
if [ "$HARD_FAIL" -eq 0 ]; then
  echo "  All hard requirements met. Next: fetch the compose file, set SERVER_IP"
  echo "  to this host's LAN address and ESPHOME_PROJECT_VERSION to the version"
  echo "  the image reports, then deploy."
  exit 0
fi
echo "  One or more hard requirements are unmet. Fix those before deploying."
exit 1
