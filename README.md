# Stock-Prediction-Model

This repository contains an advanced stock prediction system that uses a deep-learning model to predict stock prices based on historical data and technical indicators. The project is organized into separate backend and frontend folders.

**Disclaimer:** This project is a research and educational prototype. Stock prediction is inherently uncertain, and this model should not be used for real-world trading without extensive testing, validation, and risk management.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
  - [Backend Setup](#backend-setup)
  - [Frontend Setup](#frontend-setup)
- [API Endpoints](#api-endpoints)
- [Model Architecture](#model-architecture)
- [Future Improvements](#future-improvements)
- [License](#license)

## Overview

This project implements a supervised learning model using a hybrid architecture that combines:
- Convolutional layers (Conv1D) for local feature extraction,
- Bidirectional LSTM layers for capturing temporal dependencies,
- A custom attention mechanism to weigh important time steps,
- Fully connected layers to produce a final stock price prediction.

Predictions are stored in a PostgreSQL database using SQLAlchemy, and the model is exposed via a FastAPI backend. A simple HTML/CSS/JavaScript frontend is provided to interact with the API.

## Project Structure
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



