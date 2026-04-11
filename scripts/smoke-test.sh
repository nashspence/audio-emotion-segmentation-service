#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=/workspaces/emotion-diarization-service
TMP_DIR="${PROJECT_ROOT}/tests/.tmp"

cd "${PROJECT_ROOT}"

for cmd in docker curl python3 ffmpeg nvidia-smi; do
  command -v "${cmd}" >/dev/null
done

if [[ ! -f .env ]]; then
  cp .env.example .env
fi

set -a
# shellcheck disable=SC1091
source .env
set +a

API_BASE_URL="${API_BASE_URL:-http://host.docker.internal:${SERVICE_PORT}}"
mkdir -p "${TMP_DIR}"

ffmpeg -hide_banner -loglevel error -y \
  -f lavfi -i "sine=frequency=220:sample_rate=16000:duration=2" \
  -f lavfi -i "sine=frequency=660:sample_rate=16000:duration=2" \
  -filter_complex "[0:a][1:a]concat=n=2:v=0:a=1[aout]" \
  -map "[aout]" \
  -c:a libopus \
  "${TMP_DIR}/test.opus"
cp "${TMP_DIR}/test.opus" "${TMP_DIR}/test.ous"

cat > "${TMP_DIR}/diarization.json" <<'JSON'
{
  "segments": [
    {"speaker_id": "speaker_a", "start_seconds": 0.0, "end_seconds": 2.0},
    {"speaker_id": "speaker_b", "start_seconds": 2.0, "end_seconds": 4.0}
  ]
}
JSON

cleanup() {
  docker compose down -v --remove-orphans >/dev/null 2>&1 || true
}

trap cleanup EXIT

docker compose down -v --remove-orphans >/dev/null 2>&1 || true
docker compose up --build -d

container_id="$(docker compose ps -q api)"
if [[ -z "${container_id}" ]]; then
  echo "API container did not start." >&2
  exit 1
fi

max_checks="$(( STARTUP_TIMEOUT_SECONDS / 5 ))"
if [[ "${max_checks}" -lt 1 ]]; then
  max_checks=1
fi

for _ in $(seq 1 "${max_checks}"); do
  health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
  if [[ "${health}" == "healthy" ]]; then
    break
  fi
  sleep 5
done

health="$(docker inspect --format '{{if .State.Health}}{{.State.Health.Status}}{{else}}none{{end}}' "${container_id}")"
if [[ "${health}" != "healthy" ]]; then
  docker compose logs api >&2
  echo "API container did not become healthy." >&2
  exit 1
fi

python3 - <<'PY'
import json
import os
import urllib.request

with urllib.request.urlopen(f"{os.environ['API_BASE_URL']}/healthz") as response:
    payload = json.load(response)

assert payload["ready"] is True
assert payload["device"].startswith("cuda")
assert payload["models"]["categorical"].startswith("3loi/")
assert payload["models"]["avd"].startswith("3loi/")
assert "Neutral" in payload["models"]["categorical_labels"]
PY

curl -fsS \
  -F "audio=@${TMP_DIR}/test.ous;type=audio/ogg" \
  -F "diarization_file=@${TMP_DIR}/diarization.json;type=application/json" \
  "${API_BASE_URL}/v1/segment" \
  > "${TMP_DIR}/segmentation.json"

python3 - <<'PY'
import json
from pathlib import Path

payload = json.loads(Path("tests/.tmp/segmentation.json").read_text(encoding="utf-8"))

assert payload["audio"]["filename"] == "test.ous"
assert payload["audio"]["duration_seconds"] > 0
assert payload["config"]["device"].startswith("cuda")
assert isinstance(payload["window_predictions"], list) and payload["window_predictions"]
assert isinstance(payload["segments"], list) and payload["segments"]
for item in payload["window_predictions"]:
    assert item["speaker_id"] in {"speaker_a", "speaker_b"}
    assert 0 <= item["confidence"] <= 1
    assert 0 <= item["uncertainty"] <= 1
    assert len(item["categorical_probabilities"]) == 8
for item in payload["segments"]:
    assert item["speaker_id"] in {"speaker_a", "speaker_b"}
    assert item["end_seconds"] >= item["start_seconds"]
    assert len(item["categorical_probabilities"]) == 8
    assert set(item["avd_scores"]) == {"arousal", "valence", "dominance"}
PY

docker compose down -v --remove-orphans
trap - EXIT
