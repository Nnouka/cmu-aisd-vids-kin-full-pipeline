# DeepKIN Dubbing Frontend

React + Vite app for:

1. Uploading a user video
2. Starting async backend job
3. Polling job status
4. Showing translated segments
5. Muting original video and playing translated segment audio by start_time

## Local run

1. Copy .env.example to .env
2. Install packages: npm install
3. Start dev server: npm run dev

Set API URL with VITE_API_BASE_URL in .env.

## Build

- npm run build

## S3 + CloudFront deploy

After build, use:

- powershell -File scripts/deploy-s3-cloudfront.ps1 -Bucket <your-frontend-bucket> -DistributionId <your-cloudfront-id>

## Required result contract

Frontend expects backend response at GET /api/jobs/{job_id} with:

- result.results.translated_segments[]
- each segment includes start_time, end_time, transcript_rw, audio_file_url
- audio_file_name defaults to start_time with dots replaced by underscores + .wav
