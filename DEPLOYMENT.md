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

## Manual EC2 Deployment (Budget: ~$30/month)

### Instance Setup

1. **Launch EC2 Instance**
   - AMI: Ubuntu 22.04 LTS (free tier eligible or $0.0104/hr for t3.medium)
   - Instance type: `t3.medium` (2 vCPU, 4GB RAM) or `t4g.medium` (2 vCPU, 4GB RAM, cheaper ARM)
   - Storage: 50GB EBS gp3 (default is sufficient)
   - Security Group: Allow inbound on 22 (SSH), 80 (HTTP), 443 (HTTPS), 8000 (backend)
   - Key pair: Create and save your .pem file

2. **Connect via SSH**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-public-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3.12 python3-pip git docker.io docker-compose ffmpeg
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

4. **Clone Your Repository**
   ```bash
   cd /home/ubuntu
   git clone https://github.com/your-repo/deepkin-dubbing.git
   cd deepkin-dubbing
   ```

5. **Create .env File**
   ```bash
   cp backend/.env.example backend/.env
   # Edit with your AWS credentials, S3 bucket, paths
   nano backend/.env
   ```
   Required variables:
   - `AWS_ACCESS_KEY_ID`
   - `AWS_SECRET_ACCESS_KEY`
   - `AWS_REGION`
   - `INPUT_BUCKET`
   - `ARTIFACT_BUCKET`
   - `TTS_MODEL_PATH=/home/ubuntu/deepkin-dubbing/kinya_flex_tts_base_trained.pt`

6. **Build and Run Backend with Docker Compose**
   ```bash
   docker-compose up -d
   # Check logs: docker-compose logs -f backend
   ```

7. **Verify Backend is Running**
   ```bash
   curl http://localhost:8000/api/health
   curl http://localhost:8000/api/ready
   ```

8. **Setup Nginx Reverse Proxy (Optional but recommended)**
   ```bash
   sudo apt install -y nginx
   sudo nano /etc/nginx/sites-available/default
   ```
   Add:
   ```nginx
   location /api {
       proxy_pass http://localhost:8000;
       proxy_set_header Host $host;
       proxy_set_header X-Real-IP $remote_addr;
   }
   ```
   Then:
   ```bash
   sudo systemctl restart nginx
   ```

### Frontend Deployment

For the 1-week demo, you have two options:

**Option A: Serve Frontend from EC2 (Simplest)**
```bash
cd /home/ubuntu/deepkin-dubbing/frontend
npm install
npm run build
# Serve dist folder via Nginx
sudo cp -r dist/* /var/www/html/
sudo systemctl restart nginx
```

**Option B: Deploy to S3 + CloudFront (More scalable)**
- Follow the S3 + CloudFront steps from above
- Set `VITE_API_BASE_URL=http://your-ec2-ip:8000/api` in frontend .env

### Cost Breakdown ($30/month)

- **EC2 t3.medium**: ~$23/month
- **EBS Storage (50GB)**: ~$4/month
- **Data transfer out**: ~$3/month (minimal for demo)
- **Free tier offset**: If eligible, covers most of it

Total: **~$30/month**

### Monitoring & Maintenance

```bash
# Check Docker containers
docker ps

# View backend logs
docker logs -f deepkin-dubbing-backend

# Restart services
docker-compose restart

# SSH back in anytime
ssh -i your-key.pem ubuntu@your-instance-public-ip
```

### To Stop Demo (Save Costs)

```bash
# Stop instance (not terminate) - keeps EBS
aws ec2 stop-instances --instance-ids i-xxxxx --region us-east-1

# Or terminate to delete everything
aws ec2 terminate-instances --instance-ids i-xxxxx --region us-east-1
```

## CORS

Allow your frontend domain on backend CORS settings for production.
Current backend defaults to all origins for v1.

## Validation checklist

- Upload rejects files > 10 MB
- Upload rejects video > 180s
- Job status transitions to completed or failed
- result.results.translated_segments populated
- audio_file_url entries are reachable
- Playback in frontend mutes source video and plays translated audio on timestamps
