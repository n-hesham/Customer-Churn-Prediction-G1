# src/test_customer.py

import requests
import json

customers_list = [
    {
        'Age': 30, 'Tenure': 10, 'Usage Frequency': 15, 'Total Spend': 500,
        'Payment Delay': 5, 'Last Interaction': 10, 'Support Calls': 2,
        'Gender': 'Male', 'Subscription Type': 'Basic', 'Contract Length': 'Month-to-Month'
    },
    {
        'Age': 50, 'Tenure': 30, 'Usage Frequency': 5, 'Total Spend': 800,
        'Payment Delay': 20, 'Last Interaction': 5, 'Support Calls': 8,
        'Gender': 'Female', 'Subscription Type': 'Premium', 'Contract Length': 'Annual'
    },
    {
        'Age': 45, 'Tenure': 22, 'Usage Frequency': 12, 'Total Spend': 650,
        'Payment Delay': 2, 'Last Interaction': 15, 'Support Calls': 1,
        'Gender': 'Male', 'Subscription Type': 'Standard', 'Contract Length': 'Quarterly'
    },
    {
        'Age': 70, 'Tenure': 55, 'Usage Frequency': 10, 'Total Spend': 700,
        'Payment Delay': 1, 'Last Interaction': 20, 'Support Calls': 0,
        'Gender': 'Female', 'Subscription Type': 'Basic', 'Contract Length': 'Annual'
    }
]

url = "http://127.0.0.1:5001/predict"
headers = {"Content-Type": "application/json"}

print("Sending prediction requests to API (Original Logic Backend)...\n")

for i, customer_data in enumerate(customers_list):
    print(f"--- Predicting for Customer {i+1} ---")
    print(f"Input data: {customer_data}")
    try:
        # Send each customer_data as a JSON object
        response = requests.post(url, data=json.dumps(customer_data), headers=headers)

        if response.status_code == 200:
            result = response.json()
            # API returns 'probabilities_churn' as a list.
            # For a single customer input, it will be a list with one probability.
            if "probabilities_churn" in result and isinstance(result["probabilities_churn"], list) and result["probabilities_churn"]:
                churn_prob = result['probabilities_churn'][0]
                # You might also want to see the prediction (0 or 1)
                prediction = result['predictions'][0] if "predictions" in result and result["predictions"] else "N/A"
                print(f"Predicted Churn: {prediction}, Probability of Churn: {churn_prob:.4f}\n")
            else:
                print(f"API response issue: Key 'probabilities_churn' not found or invalid. Response: {result}\n")
        else:
            print(f"API Error (HTTP {response.status_code}):")
            try:
                error_details = response.json()
                print(f"  Message: {error_details.get('error', 'Unknown error details')}\n")
            except json.JSONDecodeError:
                print(f"  Response content (not JSON): {response.text}\n")

    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: Could not connect to the server at {url}.")
        print(f"Please ensure the Flask API (app.py) is running. Details: {e}\n")
        break
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {e}\n")

print("--- All prediction requests processed. ---")