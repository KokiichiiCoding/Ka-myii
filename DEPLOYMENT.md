# Ka-myii Deployment Guide 🚀

Complete guide for hosting Ka-myii in different environments.

---

## Table of Contents

1. [Local Development](#local-development)
2. [LAN Access](#lan-access)
3. [ngrok (Quick Sharing)](#ngrok-quick-sharing)
4. [Cloud Deployment](#cloud-deployment)
5. [Docker](#docker-coming-soon)
6. [Production Checklist](#production-checklist)

---

## Local Development

**Best for**: Personal use, development, testing

### Windows
```cmd
run.bat
# Choose option 1 (Production) or 2 (Demo)
```

### Linux / macOS
```bash
./run.sh
# Choose option 1 (Demo) or 2 (Full/GPU)
```

### Manual
```bash
python app.py
# Access at http://localhost:5000
```

**Pros**:
- ✅ Free
- ✅ Private
- ✅ Full control
- ✅ No setup complexity

**Cons**:
- ❌ Only accessible from your PC
- ❌ Server stops when you close terminal
- ❌ No HTTPS

---

## LAN Access

**Best for**: Sharing with family, testing on phone/tablet

### Setup

1. Start Ka-myii with `--host 0.0.0.0`:
```bash
python app.py --host 0.0.0.0 --port 5000
```

2. Find your local IP:

**Windows:**
```cmd
ipconfig
# Look for "IPv4 Address" (usually 192.168.x.x)
```

**Linux/macOS:**
```bash
ip addr show  # Linux
ifconfig      # macOS
# Look for inet address on your main interface
```

3. Access from other devices on your network:
```
http://YOUR_LOCAL_IP:5000
```

Example: `http://192.168.1.100:5000`

### Windows Firewall

If other devices can't connect, allow Python through Windows Firewall:

1. Windows Security → Firewall & network protection
2. Allow an app through firewall
3. Find Python, check **Private** (and **Public** if needed)

**Pros**:
- ✅ Easy setup
- ✅ Works on your local network
- ✅ No external services

**Cons**:
- ❌ Only works on same WiFi/LAN
- ❌ No internet access
- ❌ Firewall configuration required

---

## ngrok (Quick Sharing)

**Best for**: Temporary demos, remote testing, showing to friends

### Setup

1. Sign up at [ngrok.com](https://ngrok.com/) (free tier available)

2. Download and install ngrok:
   - **Windows**: [Download ngrok.exe](https://ngrok.com/download)
   - **Linux/macOS**: `brew install ngrok` or download from website

3. Authenticate (one-time):
```bash
ngrok authtoken YOUR_AUTH_TOKEN
```

4. Start Ka-myii locally:
```bash
python app.py
```

5. In a **new terminal**, start ngrok:
```bash
ngrok http 5000
```

6. Share the public URL (looks like `https://xxxx.ngrok.io`)

### Free Tier Limits
- ⏱️ Session expires after 2 hours
- 🔄 URL changes on restart
- 📊 40 requests/minute limit

### Paid Tier Benefits ($8/month)
- 🔗 Custom subdomain
- 🚀 No session limits
- 📈 Higher rate limits

**Pros**:
- ✅ Instant public HTTPS URL
- ✅ Works from anywhere
- ✅ No port forwarding
- ✅ Built-in analytics

**Cons**:
- ❌ Free tier has time limits
- ❌ URL changes each session (free)
- ❌ Requires ngrok account
- ❌ Shares your local machine publicly

---

## Cloud Deployment

### Hugging Face Spaces 🤗

**Best for**: Public demos, showcasing, community sharing

**Cost**: 
- Free (CPU, slow)
- ~$0.60/hour (T4 GPU, 16GB VRAM)
- Pauses when inactive

**Steps**:
1. Create account at [huggingface.co](https://huggingface.co/)
2. New Space → SDK: Gradio
3. Upload Ka-myii files
4. Add `app.py` as entry point
5. Select hardware: CPU (free) or T4 GPU (paid)

**Configuration** (`README.md` in Space):
```yaml
---
title: Ka-myii VTuber Generator
emoji: 🎨
colorFrom: purple
colorTo: blue
sdk: gradio
sdk_version: "4.0.0"
app_file: app.py
pinned: false
---
```

**Pros**:
- ✅ Free tier available
- ✅ Auto-scaling
- ✅ Built-in GPU options
- ✅ Public sharing

**Cons**:
- ❌ Limited free GPU hours
- ❌ Cold starts (slow first load)
- ❌ Public by default

---

### Railway.app 🚂

**Best for**: Always-on hobby projects, personal use

**Cost**: ~$5-10/month (CPU only), more for GPU

**Steps**:
1. Create account at [railway.app](https://railway.app/)
2. New Project → Deploy from GitHub
3. Connect your Ka-myii repository
4. Add environment variables in Railway dashboard:
```env
PORT=8080
KAMYII_MODEL_NAME=cagliostrolab/animagine-xl-3.1
```
5. Deploy!

**Pros**:
- ✅ Easy Git integration
- ✅ Auto-deploy on push
- ✅ Custom domains
- ✅ Always-on

**Cons**:
- ❌ CPU only (no GPU)
- ❌ Monthly cost
- ❌ Slower generation

---

### Replicate 🔄

**Best for**: API usage, pay-per-use, scalable

**Cost**: ~$0.0002/second GPU time (pay only when generating)

**Steps**:
1. Install Cog: `pip install cog`
2. Create `cog.yaml`:
```yaml
build:
  python_version: "3.10"
  python_packages:
    - "flask==3.0.0"
    - "torch>=2.1.0"
    - "diffusers>=0.27.0"
predict: "predict.py:Predictor"
```
3. Create `predict.py` wrapper
4. Push to Replicate: `cog push`

**Pros**:
- ✅ Pay only for usage
- ✅ Auto-scaling
- ✅ Built-in API
- ✅ No server management

**Cons**:
- ❌ Requires code adaptation
- ❌ Cold starts
- ❌ Per-request billing

---

### AWS / GCP / Azure ☁️

**Best for**: Production, enterprise, full control

**Recommended Setup**:
- **Instance**: g4dn.xlarge (AWS) or equivalent
- **GPU**: NVIDIA T4 (16GB VRAM)
- **Storage**: 50GB+ SSD
- **Cost**: ~$0.50/hour (AWS g4dn.xlarge)

**AWS EC2 Quick Start**:
1. Launch EC2 instance (Ubuntu 22.04 + Deep Learning AMI)
2. Security Group: Allow port 5000 (or 80/443)
3. SSH into instance:
```bash
ssh -i your-key.pem ubuntu@YOUR_IP
```
4. Clone Ka-myii:
```bash
git clone https://github.com/yourusername/Ka-myii.git
cd Ka-myii
```
5. Setup and run:
```bash
./run.sh
# Or for production:
pip install gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

**Pros**:
- ✅ Full control
- ✅ Scalable
- ✅ Enterprise features
- ✅ Custom configuration

**Cons**:
- ❌ Complex setup
- ❌ Server management required
- ❌ Higher cost
- ❌ Security responsibility

---

### RunPod / Vast.ai 💰

**Best for**: Cheap GPU, development, testing

**Cost**: ~$0.20-0.40/hour for 24GB VRAM GPUs

**RunPod Setup**:
1. Create account at [runpod.io](https://runpod.io/)
2. Deploy → Templates → PyTorch
3. Select GPU (RTX 3090 recommended)
4. SSH or use Jupyter
5. Clone and run Ka-myii

**Pros**:
- ✅ Very cheap GPU access
- ✅ Flexible pricing
- ✅ Good for development
- ✅ Various GPU options

**Cons**:
- ❌ Community GPUs (can be unreliable)
- ❌ Setup required
- ❌ Not always available

---

## Docker (Coming Soon)

Full Docker and docker-compose support planned for v2.1.

**Planned Features**:
- 🐳 Multi-stage builds
- 🚀 CUDA-enabled containers
- 📦 Pre-built images on Docker Hub
- 🔧 Easy environment configuration
- ⚖️ Auto-scaling with docker-compose

---

## Production Checklist

Before deploying Ka-myii publicly:

### Security
- [ ] Set strong `SECRET_KEY` in `.env`
- [ ] Add authentication (HTTP Basic Auth, OAuth)
- [ ] Rate limiting (Flask-Limiter)
- [ ] HTTPS/SSL certificate (Let's Encrypt)
- [ ] CORS configuration (whitelist domains)
- [ ] Input validation
- [ ] File upload limits (set to 16MB by default)

### Performance
- [ ] Use production WSGI server (Gunicorn, uWSGI)
- [ ] Enable response caching
- [ ] CDN for static assets
- [ ] Database for models (SQLite → PostgreSQL)
- [ ] Redis for session storage
- [ ] GPU optimization (batch processing)

### Monitoring
- [ ] Error tracking (Sentry)
- [ ] Performance monitoring (New Relic, DataDog)
- [ ] Disk space alerts
- [ ] GPU utilization monitoring
- [ ] Access logs
- [ ] Uptime monitoring

### Storage
- [ ] Automatic cleanup (old models)
- [ ] S3/Cloud storage for artifacts
- [ ] Database backups
- [ ] Model versioning

### Example Production Command
```bash
# With Gunicorn (production WSGI server)
gunicorn -w 4 \
  -b 0.0.0.0:8000 \
  --timeout 300 \
  --access-logfile logs/access.log \
  --error-logfile logs/error.log \
  app:app
```

### nginx Reverse Proxy
```nginx
server {
    listen 80;
    server_name your-domain.com;

    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for progress updates
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        # Large timeouts for generation
        proxy_read_timeout 600s;
        proxy_send_timeout 600s;
    }

    # Static files
    location /static {
        alias /path/to/Ka-myii/static;
        expires 30d;
    }
}
```

---

## Support

- 🐛 **Issues**: [GitHub Issues](https://github.com/yourusername/Ka-myii/issues)
- 📖 **Docs**: [README.md](README.md)
- 💬 **Discussions**: [GitHub Discussions](https://github.com/yourusername/Ka-myii/discussions)

---

*Ka-myii — from a prompt to a riggable VTuber, anywhere.*
