import os
import requests
from datetime import datetime

# Get the API URL from GitHub Actions environment
API_URL = os.getenv("API_URL")
if not API_URL:
    raise ValueError("API_URL environment variable not set.")

PREDICT_ENDPOINT = f"{API_URL}/predict/"

# Sample transaction for fraud prediction
payload = {
    "V1": -1.35, "V2": -0.07, "V3": 2.53, "V4": 1.37, "V5": -0.33,
    "V6": 0.46, "V7": 0.23, "V8": 0.09, "V9": 0.36, "V10": 0.09,
    "V11": -0.55, "V12": -0.61, "V13": -0.99, "V14": -0.31, "V15": 1.46,
    "V16": -0.47, "V17": 0.20, "V18": 0.02, "V19": 0.40, "V20": 0.25,
    "V21": -0.01, "V22": 0.27, "V23": -0.11, "V24": 0.06, "V25": 0.12,
    "V26": -0.18, "V27": 0.13, "V28": -0.02, "Amount": 149.62
}

report_content = f"# CML Deployment Report - Fraud Detection API 🚀\n\n"
report_content += f"Deployment Timestamp: **{datetime.utcnow().isoformat()}Z**\n\n"
report_content += f"## Deployment Status\n\n"
report_content += f"✅ Successfully deployed to GKE.\n"
report_content += f"✅ API is available at: **{API_URL}**\n\n"
report_content += f"## Live API Test\n\n"

try:
    response = requests.post(PREDICT_ENDPOINT, json=payload, timeout=15)
    response.raise_for_status()

    prediction_data = response.json()
    
    report_content += f"Sent prediction request with payload:\n"
    report_content += f"```json\n{payload}\n```\n"
    report_content += f"Received prediction response:\n"
    report_content += f"```json\n{prediction_data}\n```\n"
    report_content += f"\n**Conclusion**: The live API endpoint is responding correctly. 🎉"

except requests.exceptions.RequestException as e:
    report_content += f"🔥 **API Test Failed!**\n"
    report_content += f"Could not get a valid response from `{PREDICT_ENDPOINT}`.\n"
    report_content += f"Error: `{str(e)}`"

with open("report.md", "w") as f:
    f.write(report_content)

print("CML report 'report.md' generated successfully.")

