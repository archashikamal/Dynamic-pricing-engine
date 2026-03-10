import requests

data = {
    "product_id": "P001",
    "category": "electronics",
    "date": "2023-01-01",
    "price": 537,
    "competitor_price": 520,
    "discount_pct": 0,
    "rating": 4,
    "ad_spend": 30,
    "stock_level": 100,
    "on_promotion": 0,
    "lag_1_units": 100,
    "lag_7_units": 90,
    "goal": "Revenue"
}

response = requests.post("http://127.0.0.1:8000/optimize_price", json=data)

print("\nAPI RESPONSE:")
print(response.json())
