#!/bin/bash
# Backward-compatible evaluation entrypoint.
# The old version started three local SGLang services. The new default starts
# only the local Planner service and routes all support roles to API.

set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

export USE_API_FOR_NON_PLANNER=${USE_API_FOR_NON_PLANNER:-1}
exec bash "${SCRIPT_DIR}/eval_agentflow.sh"
