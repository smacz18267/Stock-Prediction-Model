# Stock-Prediction-Model

# Advanced Stock Prediction

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
- A custom attention mechanism to weight important time steps,
- Fully connected layers to produce a final stock price prediction.

Predictions are stored in a PostgreSQL database using SQLAlchemy, and the model is exposed via a FastAPI backend. A simple HTML/CSS/JavaScript frontend is provided to interact with the API.

## Project Structure

