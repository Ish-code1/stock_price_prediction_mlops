from pathlib import Path
import joblib
from fastapi import FastAPI
from pydantic import BaseModel, Field

MODEL_PATH = Path(__file__).resolve().parent / "models" / "stock_price_predictor.joblib"
FEATURE_NAMES = [
	"Prev Close", "Open", "High", "Low", "Last", "Close",
	"VWAP", "Volume", "Turnover", "Trades", "Deliverable Volume",
	"%Deliverble",
]
model = joblib.load(MODEL_PATH)

app = FastAPI(title="Stock Price Predictor API")

#validation of input data using Pydantic
class StockPriceInput(BaseModel):
	prev_close: float = Field(gt=0)
	open_price: float = Field(gt=0)
	high_price: float = Field(gt=0)
	low_price: float = Field(gt=0)
	last: float = Field(gt=0)
	close: float = Field(gt=0)
	vwap: float = Field(gt=0)
	volume: float = Field(ge=0)
	turnover: float = Field(ge=0)
	trades: float = Field(ge=0)
	deliverable_volume: float = Field(ge=0)
	deliverable_percent: float = Field(ge=0, le=100)

@app.get("/health") ## at url http://localhost:8000/health
def health_check():
	return{"status": "health"}

@app.get("/") # at url http://localhost:8000/
def health_check():
	return{"status": "ready to serve"}

#14/10/26

@app.post("/predict")
def predict(data: StockPriceInput):
	features = [[
		data.prev_close,
		data.open_price,
		data.high_price,
		data.low_price,
		data.last,
		data.close,
		data.vwap,
		data.volume,
		data.turnover,
		data.trades,
		data.deliverable_volume,
		data.deliverable_percent,
	]]
	prediction = model.predict(features)[0]
	return{
		"prediction" : prediction
	}   
	

#api testing/ backend testing - postman , swagger , Thunder client (extention)