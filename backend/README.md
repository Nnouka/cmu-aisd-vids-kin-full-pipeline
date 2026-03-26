# DeepKIN Dubbing Backend

FastAPI async backend for pipeline execution:

1. Upload video (max 3 minutes, 10 MB)
2. Upload to S3 via API Gateway signed URL
3. Poll transcript JSON from output bucket
4. Translate transcript segments to Kinyarwanda
5. Generate DeepKIN TTS WAV files
6. Upload timestamp-named audio files to public S3 path
7. Return result payload with results.translated_segments

## API

- GET /api/health
- GET /api/ready
- POST /api/jobs (multipart file + optional transcript_key)
- GET /api/jobs/{job_id}

Job states: queued, processing, completed, failed.

## Local run

1. Copy .env.example to .env and edit values.
2. Install dependencies:
   - pip install -r backend/requirements.txt
   - pip install -e DeepKIN-AgAI
3. Start backend:
   - uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000

## Docker run

Build image from repo root:

- docker build -f backend/Dockerfile -t deepkin-dubbing-backend .

Run container:

- docker run --rm -p 8000:8000 --env-file backend/.env -v ${PWD}/kinya_flex_tts_base_trained.pt:/app/kinya_flex_tts_base_trained.pt deepkin-dubbing-backend

For GPU host:

- docker run --rm --gpus all -p 8000:8000 --env-file backend/.env -v ${PWD}/kinya_flex_tts_base_trained.pt:/app/kinya_flex_tts_base_trained.pt deepkin-dubbing-backend

scp -i "aws-kin.pem" ~/.ssh/id_ed25519   ubuntu@ec2-52-0-99-84.compute-1.amazonaws.com:~/.ssh/id_ed25519
scp -i "aws-kin.pem" ~/.ssh/id_ed25519.pub   ubuntu@ec2-52-0-99-84.compute-1.amazonaws.com:~/.ssh/id_ed25519.pub