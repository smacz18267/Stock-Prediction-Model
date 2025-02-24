from fastapi import FastAPI, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
import datetime
from backend.database.db import engine, Base, get_db
from backend.database.models import Prediction
from backend.models import supervised_model

# Create all database tables if they do not exist
Base.metadata.create_all(bind=engine)

app = FastAPI(title="Advanced Stock Prediction API")

class PredictionRequest(BaseModel):
    ticker: str

@app.post("/predict")
def predict_stock(request: PredictionRequest, db: Session = Depends(get_db)):
    ticker = request.ticker.upper()
    try:
        # Train the advanced supervised model on the specified ticker.
        model, scaler, seq_length = supervised_model.train_supervised_model(ticker)
        predicted_price = supervised_model.predict_stock_price(model, scaler, seq_length, ticker)
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    # Save the prediction result into the PostgreSQL database.
    prediction_entry = Prediction(
        model_type="supervised",
        ticker=ticker,
        predicted_price=float(predicted_price),
        timestamp=datetime.datetime.utcnow()
    )
    db.add(prediction_entry)
    db.commit()
    db.refresh(prediction_entry)
    
    return {
        "ticker": ticker,
        "predicted_price": predicted_price,
        "timestamp": prediction_entry.timestamp.isoformat()
    }
