## Plan: FastAPI DeepKIN Video Dubbing Pipeline

Build an async FastAPI backend that reproduces the notebook pipeline (upload video -> transcript via your existing API Gateway flow -> sentence translation -> DeepKIN TTS -> timestamp-named audio uploads -> transcript enrichment) and a React frontend deployed on S3+CloudFront that plays generated audio segments against muted video using start_time scheduling. This keeps your current presigned URL/API Gateway model, adds production-safe job orchestration, and packages backend deployment in Docker.

**Steps**
1. Phase 1: Lock service contract and artifacts
2. Define API contract for async jobs: create job, get job status, get final result JSON, and list segment playback metadata. Include explicit status states (`queued`, `processing`, `completed`, `failed`) and error payload shape. *blocks phases 2-4*
3. Normalize transcript schema to `results.translated_segments` for frontend consumption. Ensure each segment includes `start_time`, `end_time`, `transcript_en`, `transcript_rw`, `audio_file_url`, and optional derived `audio_filename` where filename follows `start_time.replace('.', '_') + '.wav'`. *depends on step 2*
4. Define upload policy (3 minutes and 10 MB max) and enforce it both client-side and backend-side before processing starts. *depends on step 2*
5. Phase 2: Backend architecture (FastAPI + worker)
6. Design FastAPI endpoints: upload/initiate job, optional direct-upload helper (presigned URL passthrough), job status endpoint, and result retrieval endpoint. Keep no-auth v1 behavior as requested. *depends on phase 1*
7. Implement async processing model in backend process manager (background task queue abstraction, pluggable later for SQS/Celery). For v1, support durable job records in storage so status survives restarts. *depends on step 6*
8. Recreate notebook pipeline as modular services: transcript fetch adapter (existing API Gateway flow), sentence builder (`items_to_sentences` semantics), translation adapter (NLLB), TTS adapter (DeepKIN FlexKinyaTTS), S3 upload adapter (presigned PUT/GET). *parallelizable internally after step 7*
9. Add strict validation and limits: allowed media types, duration <= 180s, file size <= 10MB, and graceful failure messages when upstream transcript/translation/TTS fails. *depends on step 8*
10. Phase 3: Containerization and runtime hardening
11. Create Docker runtime for backend with DeepKIN install path and model mount strategy (model file externalized as volume or image artifact). Ensure GPU-aware runtime with CPU fallback policy documented. *depends on phase 2*
12. Externalize configuration to environment variables (API endpoints, bucket/prefix, model path, public artifact base URL, max limits). Remove hardcoded endpoint/bucket assumptions from service code. *parallel with step 11*
13. Add health/readiness endpoints and startup checks (model availability, required env vars, S3/API reachability). *depends on steps 11-12*
14. Provide deployment runbook for container build/run/push and production env var setup. *depends on step 13*
15. Phase 4: React frontend and playback orchestration
16. Build upload flow UI with pre-validation (max 3 min, 10 MB), submit to backend async job, and polling UX until completion/failure. *depends on phase 1 and phase 2 API completion*
17. Implement playback engine: when user presses play, mute original video and schedule audio segments from `results.translated_segments` by `start_time` (seconds float). Fetch audio from `audio_file_url`, map expected filenames by replacing `.` with `_`, and keep synchronization with seek/pause/resume handling. *depends on step 16*
18. Add transcript panel bound to current playback time and fallback behavior for missing/unavailable audio segments. *parallel with step 17*
19. Phase 5: Frontend deployment (S3 + CloudFront)
20. Configure production build artifacts upload to S3 static hosting bucket and CloudFront distribution invalidation workflow.
21. Set SPA routing behavior (custom error responses/index fallback), CORS policy to backend domain, and cache policy separation for static assets vs mutable JSON/audio metadata. *depends on step 20*
22. Document environment setup for frontend (API base URL, polling interval, timeout thresholds). *depends on step 20*
23. Phase 6: Verification and release gate
24. Run end-to-end test with one sample video under constraints and assert output JSON contract, generated audio URLs, and timestamp/file-name mapping correctness.
25. Validate frontend sync manually: play, pause, seek forward/backward, restart playback, and confirm muted source video plus aligned translated audio.
26. Execute container smoke tests (startup checks, health endpoint, model load), then perform cloud deployment smoke checks via CloudFront URL.

**Relevant files**
- c:/cmu/course-work/spring-1/ai-system-design/projects/group-work/notebooks/deployable/tts-models-deepkin.ipynb — source pipeline behavior to modularize in backend services
- c:/cmu/course-work/spring-1/ai-system-design/projects/group-work/notebooks/deployable/sample.json — example transcript item format with `start_time` and `end_time`
- c:/cmu/course-work/spring-1/ai-system-design/projects/group-work/notebooks/deployable/transcriptstandard.json — alternate transcript input format from YouTube transcript flow
- c:/cmu/course-work/spring-1/ai-system-design/projects/group-work/notebooks/deployable/install.md — DeepKIN installation baseline to convert into Docker build steps
- c:/cmu/course-work/spring-1/ai-system-design/projects/group-work/notebooks/deployable/DeepKIN-AgAI/requirements.txt — dependency source to split into minimal runtime set
- c:/cmu/course-work/spring-1/ai-system-design/projects/group-work/notebooks/deployable/DeepKIN-AgAI/deepkin/models/flex_tts.py — TTS model loading/inference reference
- c:/cmu/course-work/spring-1/ai-system-design/projects/group-work/notebooks/deployable/DeepKIN-AgAI/deepkin/production/tts_backend.py — existing inference service pattern to adapt from Flask to FastAPI

**Verification**
1. Backend contract checks: validate JSON schema for `results.translated_segments` and required keys in every segment.
2. Pipeline correctness checks: compare generated segment count and timing against source transcript sentence splits.
3. Naming checks: verify every uploaded WAV filename equals `start_time` with dots replaced by underscores.
4. Performance checks: confirm 10MB/3-minute guardrails reject invalid uploads with explicit error codes.
5. Deployment checks: run container locally with mounted model file, then test same image in cloud runtime.
6. Frontend checks: ensure muted source video + translated audio schedule remains synchronized during seek/pause/resume.

**Decisions**
- Use async jobs with status endpoint (approved).
- Standardize output contract on `translated_segments` (approved).
- Keep v1 backend unauthenticated (approved).
- Reuse existing API Gateway transcript/presigned URL integration from notebook flow (approved).
- Deployment depth: runnable package + manual deployment commands, not full IaC (approved).

**Further Considerations**
1. Job durability recommendation: use a persistent store (DynamoDB/Postgres) for job status rather than in-memory state to avoid loss on container restart.
2. Transcript source resilience: if API Gateway transcript stage fails, add a configurable fallback provider adapter in the same async job framework.
3. Cost/performance recommendation: cache translated transcript outputs by content hash so repeated video retries do not rerun translation/TTS unnecessarily.
