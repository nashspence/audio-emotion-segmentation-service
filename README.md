# emotion-diarization-service

Minimal GPU-first HTTP service for speaker-aware speech emotion segmentation on diarized audio.

## What It Does

- Accepts an uploaded audio file plus optional diarization JSON and speaker metadata.
- Runs overlapping sliding-window inference on a dedicated local GPU.
- Uses Hugging Face WavLM SER baselines:
  - `3loi/SER-Odyssey-Baseline-WavLM-Categorical`
  - `3loi/SER-Odyssey-Baseline-WavLM-Multi-Attributes`
- Smooths and merges window predictions into speaker-aware emotion segments.
- Returns JSON with raw window predictions and merged emotion segments.

## Start

```bash
cp .env.example .env
docker compose up --build
```

The container becomes healthy only after the model weights are downloaded, loaded onto the GPU, and the API is ready.

## API

`POST /v1/segment`

Multipart fields:

- `audio`: required audio file
- `diarization_json`: optional diarization JSON string
- `diarization_file`: optional diarization JSON file
- `speaker_metadata_json`: optional speaker metadata JSON object

`GET /healthz`

- Returns `200` only when the service is fully ready
- Returns `503` while weights are still downloading or loading

## Smoke Test

```bash
bash scripts/smoke-test.sh
```
