# file: .cml/cd-pipeline-fraud.py

import os
import requests
import json
from datetime import datetime

# Get the API URL from the environment variable set by the CI/CD pipeline
API_URL = os.getenv("API_URL")
REPORT_PATH = "report.md"

# --- Define a valid payload that matches the `TransactionInput` schema ---
# This is a single JSON object with all the required fields.
payload = {
    "Time": 50554.0,
    "V1": -0.798671774485497,
    "V2": 1.18509290816488,
    "V3": 0.904547185437142,
    "V4": 0.694583749252136,
    "V5": 0.219040575098198,
    "V6": -0.319295358148838,
    "V7": 0.495236343903129,
    "V8": 0.139268600382104,
    "V9": -0.760213608856806,
    "V10": 0.17054681233401,
    "V11": 0.821998442387764,
    "V12": 0.46832176679532,
    "V13": -0.0575502861155432,
    "V14": 0.573006314979332,
    "V15": 0.358687757777865,
    "V16": -0.0116327063359799,
    "V17": -0.50456991587577,
    "V18": 0.722749576258471,
    "V19": 0.861540918769592,
    "V20": -0.081298289388587,
    "V21": 0.202287281499889,
    "V22": 0.578699296051212,
    "V23": -0.0922450213734442,
    "V24": 0.0137228520784373,
    "V25": -0.246466055551771,
    "V26": -0.38005716679397,
    "V27": -0.396030194077437,
    "V28": -0.112900666069163,
    "Amount": 4.18
}

# --- Generate the CML Report ---
report_content = f"""
# CML Deployment Report - Fraud Detection API 🚀

**Deployment Timestamp:** {datetime.now().isoformat()}Z

## Deployment Status
"""

if not API_URL:
    report_content += "❌ **Deployment Failed!** Could not retrieve API URL."
else:
    report_content += f"✅ **Successfully deployed to GKE.**\n"
    report_content += f"✅ **API is available at:** `{API_URL}`\n\n"
    report_content += "## Live API Test\n"
    
    try:
        # Send the POST request with the correct payload to the /predict/ endpoint
        response = requests.post(f"{API_URL}/predict/", json=payload, timeout=20)
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        # If successful, add success details to the report
        report_content += f"✅ **API Test Passed!**\n"
        report_content += f"Status Code: `{response.status_code}`\n"
        report_content += f"``````\n"

    except requests.exceptions.HTTPError as err:
        # Handle HTTP errors (like 422, 500, etc.)
        report_content += f"🔥 **API Test Failed!**\n"
        report_content += f"Could not get a valid response from `{API_URL}/predict/`.\n"
        report_content += f"**Status Code:** `{err.response.status_code}`\n"
        report_content += f"**Reason:** `{err.response.reason}`\n"
        # Include response body, as it contains useful debug info from FastAPI
        if err.response:
            report_content += f"**Details:**\n``````\n"

    except requests.exceptions.RequestException as e:
        # Handle other request errors (like connection timeout)
        report_content += f"🔥 **API Connection Failed!**\n"
        report_content += f"Error: `{e}`\n"

# Write the final report to a markdown file
with open(REPORT_PATH, "w") as f:
    f.write(report_content)

print(f"CML report '{REPORT_PATH}' generated successfully.")


