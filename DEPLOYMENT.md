# Deployment Guide

## Architecture

- Backend: FastAPI container, async job pipeline
- Frontend: React static site on S3, distributed through CloudFront

## Backend container deployment

1. Build image
   - docker build -f backend/Dockerfile -t deepkin-dubbing-backend .
2. Push image to ECR or your registry
3. Run on ECS/Fargate, EC2, or EKS with:
   - backend listening on port 8000 internally
   - expose public traffic through a reverse proxy on ports 80/443 in production
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
   - Security Group: Allow inbound on 22 (SSH), 80 (HTTP), 443 (HTTPS)
   - Do not expose backend port 8000 publicly
   - Key pair: Create and save your .pem file

2. **Connect via SSH**
   ```bash
   ssh -i your-key.pem ubuntu@your-instance-public-ip
   ```

3. **Install Dependencies**
   ```bash
   sudo apt update && sudo apt upgrade -y
   sudo apt install -y python3.14 python3-pip git docker.io docker-compose ffmpeg
   sudo usermod -aG docker ubuntu
   newgrp docker
   ```

4. **Clone Your Repository**
   ```bash
   scp -i "aws-kin.pem" ~/.ssh/id_ed25519   ubuntu@ec2-52-0-99-84.compute-1.amazonaws.com:~/.ssh/id_ed25519
   scp -i "aws-kin.pem" ~/.ssh/id_ed25519.pub   ubuntu@ec2-52-0-99-84.compute-1.amazonaws.com:~/.ssh/id_ed25519.pub

   ssh -i "aws-kin.pem" ubuntu@ec2-52-0-99-84.compute-1.amazonaws.com
   sudo chown -R ubuntu:ubuntu ~/.ssh

   chmod 700 ~/.ssh
   chmod 600 ~/.ssh/id_ed25519
   chmod 644 ~/.ssh/id_ed25519.pub

   eval "$(ssh-agent -s)"
   ssh-add ~/.ssh/id_ed25519
   ssh -T git@github.com

   git clone git@github.com:Nnouka/cmu-aisd-vids-kin-full-pipeline.git
   cd cmu-aisd-vids-kin-full-pipeline
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

6. **Build and Run Backend with Docker Compose (private port only)**
   ```bash
   # Option A: update docker-compose to bind backend only to localhost
   # ports:
   #   - "127.0.0.1:8000:8000"

   # Option B: keep current compose file and run with an override file in prod
   docker-compose up -d
   # Check logs: docker-compose logs -f backend
   ```
   Notes:
   - SQLite is persisted via Docker volume `deepkin-backend-data` mounted at `/app/data`.
   - Avoid `docker compose down -v` or `docker-compose down -v` unless you intentionally want to delete the DB.

7. **Verify Backend is Running**
   ```bash
   curl http://localhost:8700/api/health
   curl http://localhost:8700/api/ready
   ```

8. **Setup Nginx Reverse Proxy (Required for production 80/443)**
   ```bash
   sudo apt install -y nginx
   sudo nano /etc/nginx/sites-available/default
   ```
   Add:
   ```nginx
   map $http_origin $cors_origin {
      default "";
      "http://localhost:3000" $http_origin;
      "http://localhost" $http_origin;
      "http://localhost:5173" $http_origin;
      "https://aisd-kin.yinyangr.com" $http_origin;
      "https://www.aisd-kin.yinyangr.com" $http_origin;
   }
   server {
       listen 80;
       server_name aisd-kin-api.yinyangr.com;

       client_max_body_size 15M;

       location /api/ {
         if ($request_method = OPTIONS) {
            add_header Access-Control-Allow-Origin "http://localhost:5173" always;
            add_header Access-Control-Allow-Methods "GET, POST, PUT, PATCH, DELETE, OPTIONS" always;
            add_header Access-Control-Allow-Headers "Authorization, Content-Type, Accept, Origin, X-Requested-With" always;
            add_header Access-Control-Max-Age 86400 always;
            return 204;
        }

        add_header Access-Control-Allow-Origin "http://localhost:5173" always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, PATCH, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Authorization, Content-Type, Accept, Origin, X-Requested-With" always;
           proxy_pass http://127.0.0.1:8700/api/;

           proxy_http_version 1.1;
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
        proxy_connect_timeout 60s;

           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
           proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto $scheme;
       }

       location /swagger {
           proxy_pass http://127.0.0.1:8700/swagger;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
       }

         location = /openapi.json {
            proxy_pass http://127.0.0.1:8700/openapi.json;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
         }

         # Swagger UI static assets served by FastAPI (JS/CSS)
         location ~ ^/(swagger-ui\.css|swagger-ui-bundle\.js|swagger-ui-standalone-preset\.js)$ {
            proxy_pass http://127.0.0.1:8700;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
         }

         # OAuth2 redirect endpoint used by Swagger UI
         location = /docs/oauth2-redirect {
            proxy_pass http://127.0.0.1:8700/docs/oauth2-redirect;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
            proxy_set_header X-Forwarded-Proto $scheme;
         }
   }
   ```
   Then:
   ```bash
   sudo nginx -t
   sudo systemctl restart nginx
   ```

9. **Enable HTTPS (port 443) with Let's Encrypt**
   ```bash
   sudo apt install -y certbot python3-certbot-nginx
   sudo certbot --nginx -d aisd-kin-api.yinyangr.com
   ```
   This updates Nginx to terminate TLS on 443 automatically.

10. **Verify production endpoints**
   ```bash
   curl http://api.your-domain.com/api/health
   curl https://api.your-domain.com/api/health
   curl https://api.your-domain.com/api/ready
   ```

11. **Frontend production API URL**
   Set frontend API base URL to your HTTPS domain:
   ```env
   VITE_API_BASE_URL=https://api.your-domain.com/api
   ```

### Minimal production changes checklist

- Open EC2 Security Group inbound only on 22, 80, 443.
- Keep backend service on internal port 8000 (localhost binding preferred).
- Put Nginx in front to serve public traffic on 80/443.
- Install TLS cert with Certbot for 443.
- Point frontend to `https://api.your-domain.com/api`.

### Example docker-compose production port mapping

Use this in production instead of exposing backend publicly:

```yaml
ports:
  - "127.0.0.1:8000:8000"
```

### Why this is the production pattern

- Backend remains private and cannot be reached directly from the internet.
- Nginx handles TLS termination and certificates.
- You control all public access through ports 80 and 443 only.

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
- Set `VITE_API_BASE_URL=https://api.your-domain.com/api` in frontend .env

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
