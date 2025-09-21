# Pineapple Sweetness Analysis Backend

A FastAPI-based backend service for analyzing pineapple sweetness using machine learning models. This server provides endpoints for uploading pineapple images and receiving sweetness predictions.

## Features

- **Sweetness Classification**: Analyzes uploaded pineapple images to predict sweetness levels
- **RESTful API**: FastAPI-powered endpoints for easy integration
- **Database Storage**: SQLAlchemy-based storage for prediction history
- **CORS Support**: Configured for cross-origin requests
- **File Upload**: Handles image uploads with validation
- **Machine Learning**: TensorFlow/Keras models for image classification

## Tech Stack

- **Framework**: FastAPI
- **Machine Learning**: TensorFlow, Keras
- **Database**: SQLAlchemy (SQLite)
- **Image Processing**: PIL (Pillow)
- **Server**: Uvicorn

## Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd PineappleServer
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Ensure model files are in place**
   - The ML models should be located in `assets/models/`
   - Sweetness classifier: `assets/models/pineapple_classifier/pineapple_sweetness_classifier.keras`
   - Detection model: `assets/models/pineapple_detector/best_saved_model/`

## Usage

### Starting the Server

**Option 1: Using the batch file (Windows)**
```bash
run_backend.bat
```

**Option 2: Using Python directly**
```bash
python app.py
```

**Option 3: Using Uvicorn**
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

The server will start on `http://localhost:8000`

### API Endpoints

#### Health Check
```http
GET /
```
Returns server status and information.

#### Predict Sweetness
```http
POST /predict
```
Upload a pineapple image to get sweetness prediction.

**Parameters:**
- `file`: Image file (JPG, JPEG, PNG)

**Response:**
```json
{
  "prediction_id": "uuid",
  "predicted_sweetness": "Sweet/Not Sweet",
  "confidence": 0.95,
  "timestamp": "2023-01-01T12:00:00",
  "message": "Prediction successful"
}
```

#### Get Prediction History
```http
GET /predictions
```
Retrieve all past predictions.

#### Get Specific Prediction
```http
GET /predictions/{prediction_id}
```
Get details of a specific prediction by ID.

## Project Structure

```
PineappleServer/
├── app.py                      # Main application file
├── app_backup.py              # Backup version
├── app_clean.py              # Clean version
├── requirements.txt          # Python dependencies
├── run_backend.bat          # Windows startup script
├── launch.json             # Debug configuration
├── predictions.db         # SQLite database (created automatically)
├── assets/
│   └── models/
│       ├── pineapple_classifier/
│       │   └── pineapple_sweetness_classifier.keras
│       └── pineapple_detector/
│           └── best_saved_model/
└── uploads/              # Uploaded images storage
```

## Development

### Adding New Features

1. Create a new branch for your feature
2. Make your changes
3. Test thoroughly
4. Submit a pull request

### Database Schema

The application uses SQLAlchemy with the following model:

- **Prediction**: Stores prediction results
  - `id`: Primary key
  - `prediction_id`: Unique UUID
  - `filename`: Original filename
  - `predicted_sweetness`: Prediction result
  - `confidence`: Model confidence score
  - `timestamp`: When prediction was made
  - `file_path`: Path to uploaded file

## Configuration

Key configuration options in `app.py`:

- **MODEL_PATH**: Path to the sweetness classifier model
- **UPLOAD_DIR**: Directory for uploaded files
- **MAX_FILE_SIZE**: Maximum upload file size
- **ALLOWED_EXTENSIONS**: Supported image formats

## Troubleshooting

### Common Issues

1. **Module not found errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`

2. **Model loading errors**
   - Verify model files exist in the correct paths
   - Check file permissions

3. **Database errors**
   - The SQLite database is created automatically
   - Ensure write permissions in the project directory

4. **CORS errors**
   - CORS is configured for all origins in development
   - Modify CORS settings in production as needed

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is part of a capstone project. Please respect the academic integrity guidelines.

## Deployment

### Deploying to Render

This application is configured for easy deployment on [Render](https://render.com).

#### Prerequisites
- Your code pushed to a GitHub repository
- A Render account (free tier available)

#### Deployment Steps

1. **Connect GitHub to Render**
   - Sign up/log in to [Render](https://render.com)
   - Connect your GitHub account

2. **Create a New Web Service**
   - Click "New +" → "Web Service"
   - Select your GitHub repository
   - Choose the branch to deploy (usually `main`)

3. **Configure the Service**
   - **Name**: `pineapple-backend` (or your preferred name)
   - **Environment**: `Python 3`
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn app:app --host 0.0.0.0 --port $PORT`

4. **Environment Variables** (Optional)
   Add these if you want to customize:
   - `MODEL_PATH`: Path to your ML model
   - `DETECTOR_PATH`: Path to detection model
   - `CLASS_ORDER`: Comma-separated class names

5. **Deploy**
   - Click "Create Web Service"
   - Render will automatically build and deploy your app
   - You'll get a URL like `https://your-app-name.onrender.com`

#### Using render.yaml (Alternative)

For easier configuration, this project includes a `render.yaml` file. Simply:
1. Push the `render.yaml` to your repository
2. In Render, create a new service and select "Use render.yaml"
3. Render will automatically configure everything

#### Production Considerations

- **Free Tier Limitations**: Render's free tier may spin down after inactivity
- **Model Files**: Large ML models (>500MB) might need external storage
- **Database**: Consider upgrading to PostgreSQL for production
- **Environment Variables**: Set sensitive data via Render's environment variables
- **Custom Domain**: Available on paid plans

#### Monitoring and Logs

- View logs in the Render dashboard
- Monitor performance and resource usage
- Set up alerts for downtime

## Contact

For questions or support, please open an issue in the GitHub repository.
