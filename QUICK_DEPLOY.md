# Quick Deployment Guide

## Fastest Way: Render (Recommended)

### Step 1: Prepare Your Code
1. Make sure all files are committed to Git:
   ```bash
   git add .
   git commit -m "Ready for deployment"
   ```

2. Push to GitHub/GitLab/Bitbucket:
   ```bash
   git push origin main
   ```

### Step 2: Deploy on Render

1. **Go to**: https://render.com and sign up/login

2. **Create New Web Service**:
   - Click "New +" → "Web Service"
   - Connect your Git repository
   - Select your repository

3. **Configure**:
   - **Name**: `tissue-damage-detection` (or any name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app --bind 0.0.0.0:$PORT`
   - **Plan**: Free (or choose paid for always-on)

4. **Important**: Make sure `tissue_damage_model.pth` is in your repository!

5. **Click "Create Web Service"**

6. **Wait for deployment** (takes 2-5 minutes)

7. **Your app is live!** You'll get a URL like: `https://tissue-damage-detection.onrender.com`

---

## Alternative: Railway (Also Easy)

1. **Go to**: https://railway.app and sign up

2. **New Project** → "Deploy from GitHub repo"

3. **Select your repository**

4. **Railway auto-detects** Python and uses the `Procfile`

5. **Deploy!** (Takes 2-3 minutes)

---

## Alternative: Docker (Any Platform)

If you want to deploy using Docker:

1. **Build locally** (test first):
   ```bash
   docker build -t tissue-app .
   docker run -p 5001:5001 tissue-app
   ```

2. **Deploy to any Docker platform**:
   - **Google Cloud Run**: `gcloud run deploy`
   - **AWS ECS/Fargate**: Use AWS Console
   - **Fly.io**: `fly deploy`
   - **DigitalOcean App Platform**: Connect Dockerfile

---

## Important Notes

### Model File Size
If your `tissue_damage_model.pth` is very large (>100MB):
- **Option 1**: Use Git LFS (Large File Storage)
- **Option 2**: Upload to cloud storage (S3/GCS) and download on startup
- **Option 3**: Include in repo if < 100MB (most platforms allow this)

### Environment Variables
Set these in your platform's dashboard if needed:
- `FLASK_ENV=production` (disables debug mode)
- `PORT` (usually auto-set by platform)

### Testing Locally with Production Settings
Before deploying, test with Gunicorn locally:
```bash
pip install gunicorn
gunicorn app:app --bind 0.0.0.0:5001
```

---

## Troubleshooting

**"Model file not found"**
- Check that `tissue_damage_model.pth` is committed to Git
- Verify the file path in `app.py` is correct

**"Port error"**
- Make sure you're using `$PORT` environment variable
- Don't hardcode port numbers

**"Build fails"**
- Check that all dependencies are in `requirements.txt`
- Verify Python version compatibility

**"App crashes on startup"**
- Check logs in platform dashboard
- Verify model file is accessible
- Test locally first with production settings

---

## Next Steps After Deployment

1. ✅ Test your deployed app
2. ✅ Set up custom domain (optional)
3. ✅ Configure monitoring/alerts
4. ✅ Set up CI/CD for auto-deployments

For more detailed deployment options, see `DEPLOYMENT.md`.

