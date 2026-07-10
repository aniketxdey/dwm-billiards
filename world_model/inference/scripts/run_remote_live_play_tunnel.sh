#!/usr/bin/env bash
set -euo pipefail

# Launch low-latency websocket live-play app on remote GPU and open local SSH tunnel.

INF_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd "${INF_ROOT}/.." && pwd)"

REMOTE_SSH="${REMOTE_SSH:-ubuntu@192.222.52.102}"
REMOTE_KEY_PATH="${REMOTE_KEY_PATH:-${REPO_ROOT}/bill-diff.pem}"
REMOTE_REPO="${REMOTE_REPO:-/home/ubuntu/maat}"
SYNC_CODE="${SYNC_CODE:-1}"
NO_TUNNEL="${NO_TUNNEL:-0}"

LOCAL_PORT="${LOCAL_PORT:-7863}"
REMOTE_PORT="${REMOTE_PORT:-7863}"

WM_INF_DEFAULT_TRAIN_CONFIG="${WM_INF_DEFAULT_TRAIN_CONFIG:-/home/ubuntu/maat/world_model_training/configs/dit_df_joint_v1v2v3_ctx8_2xh100_1521m_360m_resume_lr25e5.yaml}"
WM_INF_DEFAULT_CHECKPOINT="${WM_INF_DEFAULT_CHECKPOINT:-/home/ubuntu/maat/world_model_training/runs/dit_df_joint_v1v2v3_ctx8_2xh100_1521m_resume360m_20260226_181315_run01/checkpoints/ckpt_360000000.pt}"
WM_INF_DEFAULT_VAE_CHECKPOINT="${WM_INF_DEFAULT_VAE_CHECKPOINT:-/home/ubuntu/maat/VAE_training/runs/vae_60m_1xa100_20260220_204310_run01/checkpoints/ckpt_060000000.pt}"
WM_INF_DEFAULT_DATA_PATH="${WM_INF_DEFAULT_DATA_PATH:-/home/ubuntu/maat/world_model_training/manifests/joint_v1v2v3_full_400k/eval_shards.txt}"

SSH_OPTS=(-i "${REMOTE_KEY_PATH}" -o StrictHostKeyChecking=no)

if [[ "${SYNC_CODE}" == "1" ]]; then
  rsync -az --delete -e "ssh ${SSH_OPTS[*]}" \
    "${INF_ROOT}/" "${REMOTE_SSH}:${REMOTE_REPO}/world_model_inference/"
fi

STAMP="$(date -u +%Y%m%d_%H%M%S)"
REMOTE_LOG="${REMOTE_REPO}/world_model_inference/runs/live_play_${STAMP}.log"

ssh "${SSH_OPTS[@]}" "${REMOTE_SSH}" bash -s -- \
  "${REMOTE_REPO}" \
  "${REMOTE_PORT}" \
  "${REMOTE_LOG}" \
  "${WM_INF_DEFAULT_TRAIN_CONFIG}" \
  "${WM_INF_DEFAULT_CHECKPOINT}" \
  "${WM_INF_DEFAULT_VAE_CHECKPOINT}" \
  "${WM_INF_DEFAULT_DATA_PATH}" <<'EOF'
set -euo pipefail

REMOTE_REPO="$1"
REMOTE_PORT="$2"
REMOTE_LOG="$3"
WM_INF_DEFAULT_TRAIN_CONFIG="$4"
WM_INF_DEFAULT_CHECKPOINT="$5"
WM_INF_DEFAULT_VAE_CHECKPOINT="$6"
WM_INF_DEFAULT_DATA_PATH="$7"

mkdir -p "${REMOTE_REPO}/world_model_inference/runs"
cd "${REMOTE_REPO}"
export PYTHONPATH="${REMOTE_REPO}/world_model_inference/src:${REMOTE_REPO}/world_model_training/src:${REMOTE_REPO}/VAE training/src"

# Normalize the historical "VAE training" path to a no-space alias to avoid
# fragile argument/env handling through ssh command wrappers.
if [[ ! -e "${REMOTE_REPO}/VAE_training" && -d "${REMOTE_REPO}/VAE training" ]]; then
  ln -s "${REMOTE_REPO}/VAE training" "${REMOTE_REPO}/VAE_training"
fi
export WM_INF_DEFAULT_TRAIN_CONFIG
export WM_INF_DEFAULT_CHECKPOINT
export WM_INF_DEFAULT_VAE_CHECKPOINT
export WM_INF_DEFAULT_DATA_PATH

pkill -f "python3 -m world_model_inference.live_play --host 127.0.0.1 --port ${REMOTE_PORT}" || true
nohup python3 -m world_model_inference.live_play --host 127.0.0.1 --port "${REMOTE_PORT}" > "${REMOTE_LOG}" 2>&1 < /dev/null &

echo "remote_log=${REMOTE_LOG}"
EOF

# Fail early if the remote app did not come up.
if ! ssh "${SSH_OPTS[@]}" "${REMOTE_SSH}" \
  "for i in \$(seq 1 20); do \
     ss -ltnp 2>/dev/null | grep -q \"127.0.0.1:${REMOTE_PORT}\" && exit 0; \
     sleep 1; \
   done; \
   echo 'live_play_failed_to_start'; \
   test -f \"${REMOTE_LOG}\" && tail -n 120 \"${REMOTE_LOG}\" || echo 'no_remote_log_yet'; \
   exit 1"; then
  echo "Remote live-play failed to start on 127.0.0.1:${REMOTE_PORT}."
  echo "Check remote log: ${REMOTE_LOG}"
  exit 1
fi

echo "Live-play app started on remote."
echo "Open: http://127.0.0.1:${LOCAL_PORT}"
echo "Press Ctrl+C to close tunnel."

if [[ "${NO_TUNNEL}" == "1" ]]; then
  exit 0
fi

exec ssh "${SSH_OPTS[@]}" -N -L "${LOCAL_PORT}:127.0.0.1:${REMOTE_PORT}" "${REMOTE_SSH}"
