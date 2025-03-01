# Stock-Prediction-Model

This repository contains an advanced stock prediction system that uses a deep-learning model to predict stock prices based on historical data and technical indicators. The project is organized into separate backend and frontend folders.

**Disclaimer:** This project is a research and educational prototype. Stock prediction is inherently uncertain, and this model should not be used for real-world trading without extensive testing, validation, and risk management.

## Table of Contents

- Overview
- 📁 Project Structure
- Setup and Installation
  - Backend Setup
  - Frontend Setup
- 🎯API Endpoints
- 📌 Model Architecture ii
- 🔥 Future Improvements

## Overview

This project implements a supervised learning model using a hybrid architecture that combines:
- Convolutional layers (Conv1D) for local feature extraction,
- Bidirectional LSTM layers for capturing temporal dependencies,
- A custom attention mechanism to weigh important time steps,
- Fully connected layers to produce a final stock price prediction.

Predictions are stored in a PostgreSQL database using SQLAlchemy, and the model is exposed via a FastAPI backend. A simple HTML/CSS/JavaScript frontend is provided to interact with the API.

## 📁 Project Structure
```
stock-prediction-model
├── backend/
│   ├── .env                  # Environment variables (e.g., DATABASE_URL)
│   ├── app.py                # Main FastAPI application
│   ├── requirements.txt      # Python dependencies for the backend
│   ├── database/
│   │   ├── db.py             # PostgreSQL connection setup (SQLAlchemy)
│   │   └── models.py         # ORM model for storing predictions
│   ├── models/
│   │   ├── supervised_model.py   # Supervised learning model (CNN+LSTM+Attention)
│   │   └── attention_layer.py    # Custom attention layer for the model
│   ├── utils/
│   │   └── data_processing.py    # Data download, feature engineering, and scaling functions
└── frontend/
    ├── index.html            # Simple HTML interface for the stock prediction app
    ├── app.js                # JavaScript to call the backend API
    ├── style.css             # Basic styling for the frontend
```

## Setup and Installation

### **1. Clone the Repository**

```bash
git clone https://github.com/your-username/stock-prediction-model.git
cd stock-prediction-model
```

### **2. Backend Setup**

#### Activate Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

#### Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

#### Set Up PostgreSQL Database

1. Make sure you have PostgreSQL installed.
2. Create a new database:
```bash
CREATE DATABASE stock_db;
```
3. Grant privileges to your PostgreSQL user:
```bash
GRANT ALL PRIVILEGES ON DATABASE stock_db TO your_username;
```
#### Configure .env File

Create a .env file inside the backend folder:

```bash
DATABASE_URL=postgresql://your_username:your_password@localhost:5432/stock_db
```
Replace your_username and your_password with your actual PostgreSQL credentials.

#### Run the FastAPI Server
```bash
uvicorn app:app --reload
```

The server will start at http://127.0.0.1:8000, and API documentation will be available at http://127.0.0.1:8000/docs.

### **3. Frontend Setup**

#### Navigate to the Frontend Directory
```bash
cd frontend
```

#### Open the Frontend in Your Browser
You can open the index.html file directly or serve it using a local server:
```bash
python -m http.server 8080
```

Then, open http://localhost:8080 in your browser.
 
## 🎯 API Endpoints

### **POST /predict**
- Description: Predicts the next stock price for a given ticker.
- Request Body:

```bash
{
  "ticker": "AAPL"
}
```
- Response:
```bash
{
  "ticker": "AAPL",
  "predicted_price": 123.45,
  "timestamp": "2025-02-24T09:54:10.081924"
}
```
Test the API at http://127.0.0.1:8000/docs.  

## 📌 Model Architecture

This model processes historical stock data using:
- Conv1D Layers: Extracts local features from the stock price time series.
- Bidirectional LSTM Layers: Captures long-term dependencies in stock movements.
- Custom Attention Mechanism: Helps focus on relevant historical patterns.
- Dense Layers: Produces the final predicted stock price.

## 🔥 Future Improvements

- ✅ Data Enrichment: Incorporate real-world news sentiment, macroeconomic indicators, or alternative datasets.
- ✅ Advanced Model Architectures: Experiment with transformer-based models or ensemble methods.
- ✅ Hyperparameter Tuning: Implement systematic tuning with tools like Keras Tuner.
- ✅ Backtesting & Risk Management: Develop frameworks to validate the model’s effectiveness in real-world trading.
