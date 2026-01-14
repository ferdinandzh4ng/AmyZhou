# Deployment Guide

This guide covers multiple ways to deploy your Tissue Damage Detection Flask web application.

## Table of Contents

1. [Quick Deploy Options](#quick-deploy-options)
   - [Render](#render-recommended)
   - [Railway](#railway)
   - [Fly.io](#flyio)
   - [Heroku](#heroku)

2. [Docker Deployment](#docker-deployment)
   - [Local Docker](#local-docker)
   - [Docker on Cloud](#docker-on-cloud)

3. [Cloud Provider Deployment](#cloud-provider-deployment)
   - [Google Cloud Platform](#google-cloud-platform)
   - [AWS](#aws)
   - [Azure](#azure)

4. [VPS Deployment](#vps-deployment)
   - [DigitalOcean](#digitalocean)
   - [Linode](#linode)

---

## Quick Deploy Options

### Render (Recommended - Easiest)

**Pros**: Free tier available, automatic HTTPS, easy setup  
**Cons**: Free tier spins down after inactivity

#### Steps:

1. **Create a Render account**: https://render.com

2. **Prepare your repository**:
   - Push your code to GitHub/GitLab/Bitbucket
   - Make sure `tissue_damage_model.pth` is in the repo (or use external storage)

3. **Create a new Web Service**:
   - Click "New +" → "Web Service"
   - Connect your repository
   - Configure:
     - **Name**: `tissue-damage-detection`
     - **Environment**: `Python 3`
     - **Build Command**: `pip install -r requirements.txt`
     - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
     - **Plan**: Free (or paid for always-on)

4. **Add Environment Variables** (if needed):
   - `PORT`: Auto-set by Render (don't override)

5. **Deploy!** Render will automatically build and deploy your app.

**Note**: For the model file, you have two options:
- **Option A**: Include it in your repo (if < 100MB)
- **Option B**: Upload to cloud storage (S3, GCS) and download on startup

---

### Railway

**Pros**: Simple, good free tier, automatic deployments  
**Cons**: Free tier has usage limits

#### Steps:

1. **Create account**: https://railway.app

2. **Create new project**:
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository

3. **Configure**:
   - Railway auto-detects Python apps
   - Add `Procfile` (see below) or set start command:
     ```
     gunicorn app:app --bind 0.0.0.0:$PORT
     ```

4. **Deploy**: Railway automatically builds and deploys

---

### Fly.io

**Pros**: Global edge deployment, good free tier  
**Cons**: Requires CLI setup

#### Steps:

1. **Install Fly CLI**:
   ```bash
   curl -L https://fly.io/install.sh | sh
   ```

2. **Login**:
   ```bash
   fly auth login
   ```

3. **Initialize** (in your project directory):
   ```bash
   fly launch
   ```
   - Follow prompts to create app
   - Use `Dockerfile` (see Docker section below)

4. **Deploy**:
   ```bash
   fly deploy
   ```

---

### Heroku

**Pros**: Well-established, good documentation  
**Cons**: No free tier anymore (paid only)

#### Steps:

1. **Install Heroku CLI**: https://devcenter.heroku.com/articles/heroku-cli

2. **Login**:
   ```bash
   heroku login
   ```

3. **Create app**:
   ```bash
   heroku create tissue-damage-detection
   ```

4. **Add Procfile** (see below)

5. **Deploy**:
   ```bash
   git push heroku main
   ```

---

## Docker Deployment

### Local Docker

1. **Build image**:
   ```bash
   docker build -t tissue-damage-app .
   ```

2. **Run container**:
   ```bash
   docker run -p 5001:5001 tissue-damage-app
   ```

### Docker on Cloud

You can deploy the Docker image to:
- **Google Cloud Run**: Serverless, pay-per-use
- **AWS ECS/Fargate**: Container orchestration
- **Azure Container Instances**: Simple container hosting
- **DigitalOcean App Platform**: Docker support

See the `Dockerfile` in this repo for the container setup.

---

## Cloud Provider Deployment

### Google Cloud Platform

#### Option 1: Cloud Run (Recommended - Serverless)

1. **Install gcloud CLI**: https://cloud.google.com/sdk/docs/install

2. **Build and deploy**:
   ```bash
   # Build container
   gcloud builds submit --tag gcr.io/YOUR-PROJECT-ID/tissue-damage-app
   
   # Deploy to Cloud Run
   gcloud run deploy tissue-damage-app \
     --image gcr.io/YOUR-PROJECT-ID/tissue-damage-app \
     --platform managed \
     --region us-central1 \
     --allow-unauthenticated
   ```

**Cost**: Pay only for requests (very cheap, free tier available)

#### Option 2: Compute Engine (VM)

1. **Create VM**:
   ```bash
   gcloud compute instances create tissue-app-vm \
     --image-family ubuntu-2204-lts \
     --image-project ubuntu-os-cloud \
     --machine-type e2-medium
   ```

2. **SSH and setup**:
   ```bash
   gcloud compute ssh tissue-app-vm
   # Then follow VPS deployment steps below
   ```

---

### AWS

#### Option 1: Elastic Beanstalk (Easiest)

1. **Install EB CLI**:
   ```bash
   pip install awsebcli
   ```

2. **Initialize**:
   ```bash
   eb init -p python-3.11 tissue-damage-app
   ```

3. **Create and deploy**:
   ```bash
   eb create tissue-damage-env
   eb deploy
   ```

#### Option 2: EC2 (VM)

1. **Launch EC2 instance** (Ubuntu 22.04)
2. **SSH and setup** (see VPS section below)

#### Option 3: ECS/Fargate (Containers)

Use Dockerfile with AWS ECS for containerized deployment.

---

### Azure

#### Option 1: App Service (Easiest)

1. **Install Azure CLI**: https://docs.microsoft.com/en-us/cli/azure/install-azure-cli

2. **Create app**:
   ```bash
   az webapp create --resource-group myResourceGroup \
     --plan myAppServicePlan --name tissue-damage-app \
     --runtime "PYTHON:3.11"
   ```

3. **Deploy**:
   ```bash
   az webapp up --name tissue-damage-app --resource-group myResourceGroup
   ```

---

## VPS Deployment

### DigitalOcean

1. **Create Droplet**:
   - Ubuntu 22.04 LTS
   - Minimum: 2GB RAM, 1 vCPU ($12/month)
   - Recommended: 4GB RAM, 2 vCPU ($24/month)

2. **SSH into server**:
   ```bash
   ssh root@your-droplet-ip
   ```

3. **Setup**:
   ```bash
   # Update system
   apt update && apt upgrade -y
   
   # Install Python and dependencies
   apt install -y python3-pip python3-venv nginx
   
   # Clone your repo
   git clone https://github.com/yourusername/your-repo.git
   cd your-repo
   
   # Create virtual environment
   python3 -m venv venv
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt gunicorn
   
   # Test run
   gunicorn app:app --bind 0.0.0.0:8000
   ```

4. **Setup systemd service** (create `/etc/systemd/system/tissue-app.service`):
   ```ini
   [Unit]
   Description=Tissue Damage Detection App
   After=network.target

   [Service]
   User=root
   WorkingDirectory=/root/your-repo
   Environment="PATH=/root/your-repo/venv/bin"
   ExecStart=/root/your-repo/venv/bin/gunicorn app:app --bind 0.0.0.0:8000

   [Install]
   WantedBy=multi-user.target
   ```

5. **Start service**:
   ```bash
   systemctl daemon-reload
   systemctl start tissue-app
   systemctl enable tissue-app
   ```

6. **Setup Nginx** (create `/etc/nginx/sites-available/tissue-app`):
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;

       location / {
           proxy_pass http://127.0.0.1:8000;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

7. **Enable site**:
   ```bash
   ln -s /etc/nginx/sites-available/tissue-app /etc/nginx/sites-enabled/
   nginx -t
   systemctl restart nginx
   ```

8. **Setup SSL** (Let's Encrypt):
   ```bash
   apt install certbot python3-certbot-nginx
   certbot --nginx -d your-domain.com
   ```

---

### Linode

Follow the same steps as DigitalOcean (Linode uses the same Ubuntu setup).

---

## Required Files for Deployment

### Procfile (for Heroku/Railway)

Create `Procfile` in project root:
```
web: gunicorn app:app --bind 0.0.0.0:$PORT
```

### runtime.txt (optional, for Heroku)

Create `runtime.txt`:
```
python-3.11.0
```

### .dockerignore

Create `.dockerignore`:
```
__pycache__
*.pyc
*.pyo
*.pyd
.Python
venv/
env/
.git
.gitignore
README.md
*.md
.DS_Store
```

---

## Production Considerations

### 1. Use Production WSGI Server

Replace Flask's development server with Gunicorn:

```bash
pip install gunicorn
```

Start command:
```bash
gunicorn app:app --bind 0.0.0.0:$PORT --workers 2 --threads 2
```

### 2. Environment Variables

Update `app.py` to use environment variables:
```python
import os
port = int(os.environ.get('PORT', 5001))
debug = os.environ.get('FLASK_ENV') != 'production'
app.run(debug=debug, host='0.0.0.0', port=port)
```

### 3. Model File Storage

For large model files (>100MB), consider:
- **Cloud Storage**: Upload to S3/GCS/Azure Blob, download on startup
- **CDN**: Serve model from CDN
- **Git LFS**: Use Git Large File Storage

### 4. Security

- Disable debug mode in production
- Use HTTPS (most platforms provide this automatically)
- Add rate limiting (Flask-Limiter)
- Validate and sanitize all inputs (already done)

### 5. Monitoring

- Add logging
- Set up error tracking (Sentry, Rollbar)
- Monitor performance (New Relic, Datadog)

---

## Quick Start Checklist

- [ ] Choose deployment platform
- [ ] Push code to Git repository
- [ ] Add `Procfile` or update start command
- [ ] Add `gunicorn` to `requirements.txt`
- [ ] Ensure model file is accessible
- [ ] Set environment variables
- [ ] Deploy!
- [ ] Test the deployed app
- [ ] Set up custom domain (optional)
- [ ] Configure SSL/HTTPS

---

## Cost Comparison

| Platform | Free Tier | Paid Starting | Best For |
|----------|-----------|---------------|----------|
| **Render** | ✅ Yes (spins down) | $7/mo | Quick deploys |
| **Railway** | ✅ Yes (limited) | $5/mo | Simple apps |
| **Fly.io** | ✅ Yes | $0.02/hour | Global edge |
| **Heroku** | ❌ No | $7/mo | Established apps |
| **Cloud Run** | ✅ Yes | Pay-per-use | Serverless |
| **DigitalOcean** | ❌ No | $12/mo | Full control |
| **AWS/GCP/Azure** | ✅ Limited | Varies | Enterprise |

---

## Troubleshooting

### "Model file not found"
- Ensure `tissue_damage_model.pth` is in the repo or accessible
- Check file paths are relative, not absolute
- Verify file is committed to Git

### "Port already in use"
- Use `$PORT` environment variable (set by platform)
- Don't hardcode port numbers

### "Memory errors"
- Reduce Gunicorn workers: `--workers 1`
- Upgrade to larger instance/plan
- Optimize model loading (lazy load)

### "Timeout errors"
- Increase timeout in platform settings
- Optimize model inference speed
- Use GPU instances if needed

---

## Need Help?

- Check platform-specific documentation
- Review error logs in platform dashboard
- Test locally with production settings first
- Use `gunicorn` locally to test production setup

