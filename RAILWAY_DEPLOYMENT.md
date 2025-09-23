# 🚂 Railway Deployment Guide

Deploy your Pineapple Detection API to Railway with external database support.

## 🚀 Quick Deployment

### **1. Prepare Your Repository**

Make sure your repository has these files:
- ✅ `Dockerfile` - Railway will use this for building
- ✅ `railway.toml` - Railway configuration
- ✅ `requirements.txt` - Python dependencies
- ✅ `app.py` - FastAPI application
- ✅ Model files in `assets/models/`

### **2. Deploy to Railway**

#### **Option A: Connect GitHub Repository**
1. Go to [railway.app](https://railway.app)
2. Sign up/Login with GitHub
3. Click **"New Project"**
4. Select **"Deploy from GitHub repo"**
5. Choose your `pinya_server` repository
6. Railway will automatically detect the Dockerfile

#### **Option B: Deploy with Railway CLI**
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login to Railway
railway login

# Initialize project
railway init

# Deploy
railway up
```

### **3. Add PostgreSQL Database**

1. In your Railway project dashboard
2. Click **"New Service"**
3. Select **"Database"**
4. Choose **"PostgreSQL"**
5. Railway will create a PostgreSQL database

### **4. Configure Environment Variables**

In your Railway project settings, add these environment variables:

```bash
# Database (Railway will auto-set DATABASE_URL)
DATABASE_TYPE=postgresql

# Models
MODEL_PATH=/app/assets/models/pineapple_classifier/pineapple_sweetness_classifier.keras
DETECTOR_PATH=/app/assets/models/pineapple_detector
CLASS_ORDER=High,Medium,Low

# Optional
DEBUG=false
```

**Note**: Railway automatically sets `DATABASE_URL` when you add a PostgreSQL service.

### **5. Deploy and Test**

1. Railway will automatically build and deploy
2. Check the deployment logs
3. Visit your Railway URL (e.g., `https://your-app.railway.app`)
4. Test the health endpoint: `https://your-app.railway.app/health`

## 🔧 Railway Configuration

### **railway.toml**
```toml
[build]
builder = "DOCKERFILE"

[deploy]
healthcheckPath = "/health"
healthcheckTimeout = 300
restartPolicyType = "ON_FAILURE"
restartPolicyMaxRetries = 10

[env]
DATABASE_TYPE = "postgresql"
MODEL_PATH = "/app/assets/models/pineapple_classifier/pineapple_sweetness_classifier.keras"
DETECTOR_PATH = "/app/assets/models/pineapple_detector"
CLASS_ORDER = "High,Medium,Low"
```

### **Dockerfile Features**
- Uses Railway's `PORT` environment variable
- Optimized for Railway's build process
- Health checks configured
- All system dependencies included

## 📊 Railway Benefits

### **✅ Advantages**
- **Free Tier**: $5 credit monthly
- **PostgreSQL Database**: Free tier included
- **Automatic Deployments**: Git-based deployments
- **Custom Domains**: Free subdomain
- **Environment Variables**: Easy configuration
- **Logs**: Real-time deployment logs
- **Metrics**: Performance monitoring

### **📈 Scaling**
- **Hobby Plan**: $5/month - Good for personal projects
- **Pro Plan**: $20/month - For production apps
- **Team Plan**: $99/month - For teams

## 🧪 Testing Your Deployment

### **Health Check**
```bash
curl https://your-app.railway.app/health
```

### **API Documentation**
Visit: `https://your-app.railway.app/docs`

### **Test Prediction**
```bash
curl -X POST "https://your-app.railway.app/predict" \
     -H "Content-Type: multipart/form-data" \
     -F "file=@test_image.jpg"
```

### **Get Predictions History**
```bash
curl https://your-app.railway.app/predictions
```

## 🔍 Troubleshooting

### **Build Issues**
- Check Railway logs for build errors
- Ensure all model files are in the repository
- Verify `requirements.txt` has all dependencies

### **Database Issues**
- Ensure PostgreSQL service is running
- Check `DATABASE_URL` environment variable
- Verify database connection in logs

### **Memory Issues**
- Railway free tier has memory limits
- Models use lazy loading to save memory
- Consider upgrading if needed

### **Common Errors**
- **Build timeout**: Optimize Dockerfile layers
- **Memory exceeded**: Use lazy loading (already implemented)
- **Database connection failed**: Check DATABASE_URL

## 🚀 Production Tips

### **Performance Optimization**
1. **Lazy Loading**: Models load only when needed
2. **Memory Management**: Models unload after predictions
3. **Database Indexing**: Add indexes for better performance
4. **Caching**: Consider Redis for caching

### **Monitoring**
- Use Railway's built-in metrics
- Set up alerts for errors
- Monitor database performance
- Track API response times

### **Security**
- Set `DEBUG=false` in production
- Use HTTPS (Railway provides automatically)
- Validate all inputs
- Rate limiting (consider adding)

## 📱 Frontend Integration

Your frontend can now use the Railway URL:

```javascript
// Example API call
const response = await fetch('https://your-app.railway.app/predict', {
  method: 'POST',
  body: formData
});

const result = await response.json();
console.log(result);
```

## 🎯 Next Steps

1. **Deploy to Railway** using the steps above
2. **Test all endpoints** to ensure everything works
3. **Set up monitoring** and alerts
4. **Configure custom domain** if needed
5. **Scale up** when ready for production

Your Pineapple API is now Railway-ready! 🚂🍍
