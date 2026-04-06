
#!/usr/bin/env bash
set -euo pipefail
RUN_ID="full_v2diverse_20260222_154743"
S3_DATA_PREFIX="s3://videogen-pool-v2-237586137680/dataraw_v2/full_v2diverse_20260222_154743"
S3_LOG_PREFIX="s3://videogen-pool-v2-237586137680/ops/runs/full_v2diverse_20260222_154743"
ARTIFACT_URI="s3://videogen-pool-v2-237586137680/ops/artifacts/full_v2diverse_20260222_154743/data_generation_package_v2.tar.gz"
EPISODES=100000
SHARD_SIZE=100
FRAMES=600
WORKERS=64
SEED=4242
RUN_ROOT="/opt/neural_pool/${RUN_ID}"
CODE_TAR="${RUN_ROOT}/data_generation_package.tar.gz"
LOG_FILE="${RUN_ROOT}/generate.log"
PIP_LOG="${RUN_ROOT}/pip_install.log"
SUMMARY_JSON="${RUN_ROOT}/summary.json"
mkdir -p "${RUN_ROOT}"
START_TS=$(date -u +%s)
START_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
sudo dnf -y install python3 python3-pip tar >/dev/null 2>&1 || true
aws --version > "${RUN_ROOT}/aws_version.txt" 2>&1 || true
python3 --version > "${RUN_ROOT}/python_version.txt" 2>&1 || true
aws s3 cp "${ARTIFACT_URI}" "${CODE_TAR}"
tar -xzf "${CODE_TAR}" -C "${RUN_ROOT}"
cd "${RUN_ROOT}/data_generation_package"
rm -rf "${RUN_ROOT}/venv"
python3 -m venv "${RUN_ROOT}/venv" > "${PIP_LOG}" 2>&1
source "${RUN_ROOT}/venv/bin/activate"
python -m pip install --upgrade pip >> "${PIP_LOG}" 2>&1
python -m pip install -r requirements.txt >> "${PIP_LOG}" 2>&1
(
  while true; do
    aws s3 cp "${LOG_FILE}" "${S3_LOG_PREFIX}/logs/generate.partial.log" >/dev/null 2>&1 || true
    sleep 60
  done
) &
SYNC_PID=$!
set +e
python generate_pool_shards.py   --episodes "${EPISODES}"   --shard-size "${SHARD_SIZE}"   --frames "${FRAMES}"   --workers "${WORKERS}"   --seed "${SEED}"   --s3-prefix "${S3_DATA_PREFIX}"   --output-dir "${RUN_ROOT}/out"   2>&1 | tee "${LOG_FILE}"
GEN_EXIT=${PIPESTATUS[0]}
set -e
kill ${SYNC_PID} >/dev/null 2>&1 || true
END_TS=$(date -u +%s)
END_ISO=$(date -u +%Y-%m-%dT%H:%M:%SZ)
DURATION_SEC=$((END_TS - START_TS))
SHARD_INFO=$(aws s3 ls "${S3_DATA_PREFIX}/raw/shards/" --recursive 2>/dev/null || true)
SHARD_COUNT=$(printf "%s
" "${SHARD_INFO}" | awk 'NF>=4 {c++} END{print c+0}')
SHARD_BYTES=$(printf "%s
" "${SHARD_INFO}" | awk 'NF>=4 {b+=$3} END{print b+0}')
python3 - <<PY2
import json
summary = {
  "run_id": "${RUN_ID}",
  "start_utc": "${START_ISO}",
  "end_utc": "${END_ISO}",
  "duration_sec": ${DURATION_SEC},
  "generator_exit_code": ${GEN_EXIT},
  "episodes": ${EPISODES},
  "shard_size": ${SHARD_SIZE},
  "frames": ${FRAMES},
  "workers": ${WORKERS},
  "seed": ${SEED},
  "s3_data_prefix": "${S3_DATA_PREFIX}",
  "s3_log_prefix": "${S3_LOG_PREFIX}",
  "shards_uploaded": int(${SHARD_COUNT}),
  "shard_bytes": int(${SHARD_BYTES}),
  "generator_variant": "data_generation_package_v2_diverse"
}
with open("${SUMMARY_JSON}", "w") as f:
    json.dump(summary, f, indent=2)
PY2
aws s3 cp "${LOG_FILE}" "${S3_LOG_PREFIX}/logs/generate.log"
aws s3 cp "${PIP_LOG}" "${S3_LOG_PREFIX}/logs/pip_install.log"
aws s3 cp "${RUN_ROOT}/aws_version.txt" "${S3_LOG_PREFIX}/logs/aws_version.txt"
aws s3 cp "${RUN_ROOT}/python_version.txt" "${S3_LOG_PREFIX}/logs/python_version.txt"
aws s3 cp "${SUMMARY_JSON}" "${S3_LOG_PREFIX}/summary.json"
if [ ${GEN_EXIT} -ne 0 ]; then
  exit ${GEN_EXIT}
fi
echo "Full run complete: ${RUN_ID}"
