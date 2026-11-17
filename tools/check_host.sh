#!/usr/bin/env bash
# Pre-flight check for the machine that will run the benchmark.
#
#   ./tools/check_host.sh [--dir /path/with/space]
#
# Verifies the parts that are cheap to check and expensive to discover halfway
# through a 40-minute Isaac Sim build: driver, container GPU access, compose,
# free disk, and whether the contest base image is already pulled.
set -uo pipefail

TARGET_DIR="${HOME}"
if [[ "${1:-}" == "--dir" && -n "${2:-}" ]]; then
  TARGET_DIR="$2"
fi

failures=0
row() { printf '%-30s %s\n' "$1" "$2"; }
ok()   { row "$1" "ok${2:+ ($2)}"; }
bad()  { row "$1" "FAIL: $2"; failures=$((failures + 1)); }
warn() { row "$1" "WARN: $2"; }

echo "host pre-flight"
echo "---------------"

# ── tooling ────────────────────────────────────────────────────────────────
for tool in git docker awk df; do
  if command -v "$tool" >/dev/null 2>&1; then ok "$tool" "$(command -v "$tool")"; else bad "$tool" "not installed"; fi
done

if docker compose version >/dev/null 2>&1; then
  ok "docker compose" "$(docker compose version --short 2>/dev/null)"
else
  bad "docker compose" "the v2 plugin is required by the simulator Makefile"
fi

# ── GPU ────────────────────────────────────────────────────────────────────
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu="$(nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | head -1)"
  if [[ -n "$gpu" ]]; then ok "GPU" "$gpu"; else bad "GPU" "nvidia-smi reports no device"; fi
else
  bad "GPU" "nvidia-smi not found; install the NVIDIA driver first"
fi

if command -v docker >/dev/null 2>&1; then
  if docker info >/dev/null 2>&1; then
    info="$(docker info --format '{{.ServerVersion}}')"
    ok "docker daemon" "$info"
    if docker info --format '{{json .Runtimes}}' 2>/dev/null | grep -q nvidia; then
      ok "nvidia runtime" "registered"
    else
      warn "nvidia runtime" "not registered; 'docker run --gpus all' will fail without nvidia-container-toolkit"
    fi
    if docker image inspect aij-green-challenge.cr.cloud.ru/green-vla:latest >/dev/null 2>&1; then
      digest="$(docker image inspect --format '{{index .RepoDigests 0}}' aij-green-challenge.cr.cloud.ru/green-vla:latest 2>/dev/null)"
      ok "contest base image" "$digest"
    else
      warn "contest base image" "not pulled yet (docker pull aij-green-challenge.cr.cloud.ru/green-vla:latest)"
    fi
  else
    warn "docker daemon" "not reachable as this user (add yourself to the docker group)"
  fi
fi

# ── disk ───────────────────────────────────────────────────────────────────
if [[ -d "$TARGET_DIR" ]]; then
  available_kb="$(df -Pk "$TARGET_DIR" | awk 'NR==2 {print $4}')"
  available_gb=$((available_kb / 1024 / 1024))
  if (( available_gb >= 200 )); then
    ok "free disk ($TARGET_DIR)" "${available_gb} GB"
  elif (( available_gb >= 100 )); then
    warn "free disk ($TARGET_DIR)" "${available_gb} GB — tight; Isaac Sim + assets + datasets want 150-250 GB"
  else
    bad "free disk ($TARGET_DIR)" "${available_gb} GB — need at least ~150 GB"
  fi
else
  warn "free disk" "$TARGET_DIR does not exist"
fi

# ── memory ─────────────────────────────────────────────────────────────────
if [[ -r /proc/meminfo ]]; then
  mem_gb=$(( $(awk '/MemTotal/ {print $2}' /proc/meminfo) / 1024 / 1024 ))
  if (( mem_gb >= 32 )); then ok "RAM" "${mem_gb} GB"; else warn "RAM" "${mem_gb} GB — 32 GB+ recommended"; fi
fi

echo "---------------"
if (( failures > 0 )); then
  echo "${failures} blocking problem(s); see docs/COMMANDS.md"
  exit 1
fi
echo "ready. Next: docs/COMMANDS.md, section 2"
