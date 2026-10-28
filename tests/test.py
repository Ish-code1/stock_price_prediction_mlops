#Import TestClient so we can send simulated HTTP requests directly to FASTAPI

from fastapi.testclient import TestClient 
#import the application object defined in main.py
from main import app

#create a reusable test client for all test function
client = TestClient(app)

def test_health_endpoint():
    """
    Verify that the application is running and can respond to a health-check request
    """
    response = client.get("/health")
    assert response.status_code == 200 # a successfull http request shd return status code 200

    #now check rsponse body
    assert response.json() == {"status": "health"}

def test_valid_stock_price_prediction():
    """
        Verify that valid stock price measurements produce a prediction
        """
    payload = {
        "prev_close": 100.0,
        "open_price": 101.0,
        "high_price": 102.0,
        "low_price": 99.0,
        "last": 101.0,
        "close": 100.5,
        "vwap": 100.2,
        "volume": 100000.0,
        "turnover": 10000000.0,
        "trades": 1000.0,
        "deliverable_volume": 50000.0,
        "deliverable_percent": 50.0,
    }
    #send a simulated http post request with json input
    response = client.post("/predict",json=payload)
    assert response.status_code ==200
    #convert the json response into python dictionary
    result = response.json()

    #check the response contract: both fields must exist
    assert "prediction" in result

def test_invalid_stock_price_input():
     payload = {
            "prev_close": -100.0,
            "open_price": 101.0,
            "high_price": 102.0,
            "low_price": 99.0,
            "last": 101.0,
            "close": 100.5,
            "vwap": 100.2,
            "volume": 100000.0,
            "turnover": 10000000.0,
            "trades": 1000.0,
            "deliverable_volume": 50000.0,
            "deliverable_percent": 50.0,
        }
     #send a simulated http post request with json input
     response = client.post("/predict",json=payload)
     #fastapi automatically returns a HTTP 422 status code for invalid input data
     assert response.status_code == 422
