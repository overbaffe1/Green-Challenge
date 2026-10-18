#!/usr/bin/env bash
# Run the GreenSim multi-scenario benchmark against a policy server.
#
#   GREEN_SIM_DIR=~/green_challenge ./eval/run_benchmark.sh
#
# Environment:
#   GREEN_SIM_DIR  path to the GreenSim checkout (required)
#   POLICY_PORT    public policy port (default 8999)
#   RUN_PATH       run config (default assets/runs/green_challenge.json)
#   RESULTS_PATH   where to copy results.json (default: ./results.<timestamp>.json)
#   LIVESTREAM     pass through to the simulator (0/1/2, default 0)
set -euo pipefail

GREEN_SIM_DIR="${GREEN_SIM_DIR:-${1:-}}"
if [[ -z "${GREEN_SIM_DIR}" || ! -d "${GREEN_SIM_DIR}/scripts" ]]; then
  echo "usage: GREEN_SIM_DIR=/path/to/green_challenge $0" >&2
  exit 2
fi

POLICY_PORT="${POLICY_PORT:-8999}"
RUN_PATH="${RUN_PATH:-assets/runs/green_challenge.json}"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
RESULTS_PATH="${RESULTS_PATH:-$PWD/results.${STAMP}.json}"

echo "policy port : ${POLICY_PORT}"
echo "run config  : ${RUN_PATH}"
echo "results     : ${RESULTS_PATH}"

# The simulator container uses network_mode: host, so the policy server must be
# reachable on 127.0.0.1:${POLICY_PORT} on this host.
( cd "${GREEN_SIM_DIR}" && \
  LIVESTREAM="${LIVESTREAM:-0}" \
  POLICY_PORT="${POLICY_PORT}" \
  RUN_PATH="${RUN_PATH}" \
  RESULTS_PATH="${RESULTS_PATH}" \
  RUN_BENCHMARK=1 \
  make up )

echo
echo "results written to ${RESULTS_PATH}"
echo "next: python eval/report.py ${RESULTS_PATH} --sweep"
