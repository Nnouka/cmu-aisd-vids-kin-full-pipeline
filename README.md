# DeepKIN Dubbing (Local Run Guide)

This repo includes:
- Backend: FastAPI pipeline service in `backend/`
- Frontend: React + Vite app in `frontend/`
- DeepKIN source package in `DeepKIN-AgAI/`

## 1. Create Python Environment

```sh
conda create -n deepkin python=3.12 -y
conda activate deepkin
```
or

```sh
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa -y
sudo apt update
sudo apt install -y python3.14 python3.14-venv python3.14-dev

cd /home/ubuntu/your-project
rm -rf .venv
python3.14 -m venv .venv
source .venv/bin/activate
python -V

python -m pip install --upgrade pip
```

## 2. Install Backend + DeepKIN Dependencies

From the repository root:

```sh
pip install -r backend/requirements.txt
pip install -e DeepKIN-AgAI
```

## 3. Configure Backend Environment

```sh
cp backend/.env.example backend/.env
```

Windows PowerShell alternative:

```powershell
Copy-Item backend\.env.example backend\.env
```

Edit `backend/.env` and confirm these values are correct for your environment:
- `GENERATE_SIGNED_URL_ENDPOINT`
- `INPUT_BUCKET`
- `OUTPUT_BUCKET`
- `ARTIFACT_BUCKET`
- `PUBLIC_PREFIX`
- `PUBLIC_ARTIFACT_STORE`
- `TTS_MODEL_PATH`

## 4. Download `kinya_flex_tts_base_trained.pt`

A standalone script was added at `backend/scripts/download_tts_model.py`.

From repo root:

```sh
python backend/scripts/download_tts_model.py --output kinya_flex_tts_base_trained.pt
```

Optional flags:

```sh
python backend/scripts/download_tts_model.py --bucket cmu-aisd-output --key kinya_flex_tts_base_trained.pt --output backend/kinya_flex_tts_base_trained.pt --force
```

If you saved the model in a different location, set `TTS_MODEL_PATH` in `backend/.env` accordingly.

## 5. Run Backend Locally

From repo root:

```sh
uvicorn app.main:app --app-dir backend --reload --host 0.0.0.0 --port 8000
```

Verify:
- `http://localhost:8000/api/health`
- `http://localhost:8000/api/ready`
- Swagger: `http://localhost:8000/swagger`

## 6. Run Frontend Locally

In a new terminal:

```sh
cd frontend
npm install
```

Create frontend env file:

```sh
cp .env.example .env
```

Windows PowerShell alternative:

```powershell
Copy-Item .env.example .env
```

Set API base in `frontend/.env`:

```env
VITE_API_BASE_URL=http://localhost:8000/api
```

Start frontend:

```sh
npm run dev
```

Open the URL shown by Vite (typically `http://localhost:5173`).

## 7. Optional Docker Backend Run

From repo root:

```sh
docker build -f backend/Dockerfile -t deepkin-dubbing-backend .
docker run --rm -p 8000:8000 --env-file backend/.env -v ${PWD}/kinya_flex_tts_base_trained.pt:/app/kinya_flex_tts_base_trained.pt deepkin-dubbing-backend
```
