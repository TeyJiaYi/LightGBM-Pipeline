import requests

url = "http://localhost:8001/predict"
payload = {
    "payFrequency": 2,
    "apr": 0.35,
    "nPaidOff": 1,
    "loanAmount": 1200.0,
    "originallyScheduledPaymentAmount": 300,
    "leadType": 1,
    "leadCost": 25,
    "hasCF": 1,
    "region_code": 101
}

res = requests.post(url, json=payload)
print("Status:", res.status_code)
print("Response:", res.json())
