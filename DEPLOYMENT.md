# Deployment Guide

## Architecture

- Backend: FastAPI container, async job pipeline
- Frontend: React static site on S3, distributed through CloudFront

## Backend container deployment

1. Build image
   - docker build -f backend/Dockerfile -t deepkin-dubbing-backend .
2. Push image to ECR or your registry
3. Run on ECS/Fargate, EC2, or EKS with:
   - port 8000 exposed
   - backend/.env values configured as environment variables
   - model file mounted or baked at TTS_MODEL_PATH
4. Verify endpoints:
   - /api/health
   - /api/ready

## Frontend deployment to S3 + CloudFront

1. In frontend directory:
   - npm install
   - npm run build
2. Upload dist assets to S3 bucket
3. Configure CloudFront origin to that bucket
4. Configure SPA fallback:
   - 403 and 404 -> /index.html
5. Invalidate CloudFront cache after each deploy

## CORS

Allow your frontend CloudFront domain on backend CORS settings for production.
Current backend defaults to all origins for v1.

## Validation checklist

- Upload rejects files > 10 MB
- Upload rejects video > 180s
- Job status transitions to completed or failed
- result.results.translated_segments populated
- audio_file_url entries are reachable
- Playback in frontend mutes source video and plays translated audio on timestamps
