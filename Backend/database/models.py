from sqlalchemy import Column, Integer, Float, String, DateTime
from .db import Base
import datetime

class Prediction(Base):
    __tablename__ = "predictions"
    
    id = Column(Integer, primary_key=True, index=True)
    model_type = Column(String, index=True) 
    ticker = Column(String, index=True)
    predicted_price = Column(Float)
    actual_price = Column(Float, nullable=True)
    timestamp = Column(DateTime, default=datetime.datetime.utcnow)
