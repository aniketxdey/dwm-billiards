#!/usr/bin/env bash
set -euo pipefail

WM_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export CONFIG_PATH="${CONFIG_PATH:-${WM_ROOT}/configs/dit_df_ctx8_2xh100_resume_120m.yaml}"
exec "${WM_ROOT}/scripts/run_train_2xh100.sh"
