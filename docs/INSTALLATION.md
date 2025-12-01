# Installation & Setup Guide

## Quick Start Guide for AstroGuard

This guide will help you set up and run the complete AstroGuard system.

---

## Prerequisites

Before starting, ensure you have:

- **Python 3.9+** installed
- **Node.js 18+** and npm
- **Git** for version control
- **8 GB RAM** minimum
- **2 GB free disk space**

### Check Prerequisites

```bash
python --version    # Should be 3.9 or higher
node --version      # Should be 18.0 or higher
npm --version       # Should be 9.0 or higher
```

---

## Step 1: Clone Repository

```bash
git clone https://github.com/TanmayCJ/Asteroid-Collision-Detector.git
cd Asteroid-Collision-Detector
```

---

## Step 2: Set Up Machine Learning Module

### Install Python Dependencies

```bash
cd ml
pip install -r requirements.txt
```

### Generate Training Data

```bash
python preprocessing.py
```

This creates synthetic satellite data in `ml/data/synthetic_tle.json`.

### Train the Model

```bash
python train.py
```

**Expected output:**
```
Generating synthetic satellite constellation...
✓ Generated 50 satellites
Processing satellite pairs...
✓ Prepared training data
Training Collision Prediction Model
...
✓ Training complete!
Model saved to: ml/models/collision_predictor.h5
```

**Training time:** 5-10 minutes on CPU

**Files created:**
- `ml/models/collision_predictor.h5` - Trained LSTM model
- `ml/models/feature_scaler.pkl` - Feature normalization
- `ml/outputs/training_history.json` - Training metrics
- `ml/outputs/training_history.png` - Training curves

### Test Predictions (Optional)

```bash
python predict.py
```

This runs demo predictions on test satellite pairs.

---

## Step 3: Set Up Backend API

### Install Backend Dependencies

```bash
cd ../backend
pip install -r requirements.txt
```

### Configure Environment (Optional)

Create `.env` file:
```env
API_HOST=0.0.0.0
API_PORT=8000
MODEL_PATH=../ml/models/collision_predictor.h5
SCALER_PATH=../ml/models/feature_scaler.pkl
```

### Start the Backend Server

```bash
uvicorn main:app --reload --port 8000
```

**Expected output:**
```
🚀 AstroGuard API Starting...
✓ Database initialized
✓ ML model loaded
✓ API ready at http://localhost:8000
✓ Documentation at http://localhost:8000/docs
```

### Test the API

Open browser to: `http://localhost:8000/docs`

You should see interactive API documentation (Swagger UI).

**Test endpoints:**
- `GET /health` - Check system status
- `GET /objects` - List satellites
- `GET /predict?objectA=SAT-0001&objectB=SAT-0002` - Run prediction

---

## Step 4: Set Up Frontend

### Install Node Dependencies

```bash
cd ../frontend
npm install
```

This installs:
- Next.js
- React
- Three.js
- Tailwind CSS
- All other dependencies

### Configure API URL (Optional)

Create `.env.local`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Start Development Server

```bash
npm run dev
```

**Expected output:**
```
- ready started server on 0.0.0.0:3000, url: http://localhost:3000
- event compiled client and server successfully
✓ Ready in 2.5s
```

### Access the Application

Open browser to: `http://localhost:3000`

You should see the AstroGuard dashboard with:
- Satellite selection dropdowns
- 3D orbit visualization
- Prediction controls

---

## Step 5: Run Your First Prediction

1. **Select Satellites**
   - Satellite A: Choose any satellite from dropdown
   - Satellite B: Choose a different satellite

2. **Run Prediction**
   - Click "Run Prediction" button
   - Wait 1-2 seconds for analysis

3. **View Results**
   - Risk gauge shows risk level
   - 3D visualization displays orbits
   - Alert card provides recommendations
   - Timeline slider shows risk evolution

---

## Troubleshooting

### Problem: ML model not found

**Solution:**
```bash
cd ml
python train.py
```

### Problem: Backend won't start - "ML model not loaded"

**Solution:**
Ensure model is trained first:
```bash
cd ml
python train.py
cd ../backend
uvicorn main:app --reload
```

### Problem: Frontend can't connect to backend

**Solution:**
1. Verify backend is running on port 8000
2. Check `.env.local` has correct API URL
3. Clear browser cache and reload

### Problem: Port already in use

**Backend:**
```bash
# Use different port
uvicorn main:app --reload --port 8001
```

**Frontend:**
```bash
# Use different port
npm run dev -- -p 3001
```

### Problem: CORS errors in browser

**Solution:**
Backend CORS is configured for `http://localhost:3000`. If using different port, update `backend/main.py`:

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Add your port
    ...
)
```

---

## Production Deployment

### Build for Production

**Frontend:**
```bash
cd frontend
npm run build
npm run start
```

**Backend:**
```bash
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Using Docker (Optional)

Create `docker-compose.yml`:
```yaml
version: '3.8'
services:
  backend:
    build: ./backend
    ports:
      - "8000:8000"
    volumes:
      - ./ml/models:/app/ml/models
  
  frontend:
    build: ./frontend
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://backend:8000
```

Run:
```bash
docker-compose up -d
```

---

## System Requirements

### Minimum Requirements
- **CPU:** 2 cores
- **RAM:** 8 GB
- **Storage:** 2 GB
- **OS:** Windows 10, macOS 11+, Linux

### Recommended Requirements
- **CPU:** 4+ cores
- **RAM:** 16 GB
- **Storage:** 5 GB SSD
- **GPU:** NVIDIA GPU for faster training (optional)

---

## Development Tools

### Recommended IDE
- **VS Code** with extensions:
  - Python
  - Pylance
  - ES7+ React snippets
  - Tailwind CSS IntelliSense

### Code Formatting
```bash
# Python
pip install black
black ml/ backend/

# JavaScript/TypeScript
cd frontend
npm run lint
```

---

## Getting Help

### Documentation
- **ML Model**: See `docs/ML_MODEL.md`
- **API**: See `http://localhost:8000/docs`
- **Project Abstract**: See `docs/PROJECT_ABSTRACT.md`

### Common Issues
- Check Python version: `python --version`
- Check Node version: `node --version`
- Verify all dependencies installed
- Ensure model is trained before starting backend

### Support
- **GitHub Issues**: Report bugs
- **Email**: contact@astroguard.space

---

## Next Steps

After successful setup:

1. **Explore the Notebook**
   ```bash
   cd ml/notebooks
   jupyter notebook training.ipynb
   ```

2. **Test API Endpoints**
   - Use Swagger UI at `http://localhost:8000/docs`
   - Try different satellite pairs

3. **Customize Frontend**
   - Edit components in `frontend/src/components/`
   - Modify styles in `frontend/tailwind.config.js`

4. **Experiment with Model**
   - Adjust hyperparameters in `ml/train.py`
   - Try different architectures
   - Retrain with more data

---

## Success Checklist

- [ ] Python 3.9+ installed
- [ ] Node.js 18+ installed
- [ ] ML dependencies installed
- [ ] Model trained successfully
- [ ] Backend running on port 8000
- [ ] Frontend running on port 3000
- [ ] Can access API docs at /docs
- [ ] Can run predictions in frontend
- [ ] 3D visualization working

---

**Congratulations!** Your AstroGuard system is now fully operational. 🚀
