# create_test_event.py
import base64
import json

# Read your real PDF file
with open('events/Minfy_AI_Cloud_Report_Q3_2025.pdf', 'rb') as f:
    pdf_bytes = f.read()

# Convert to base64
base64_content = base64.b64encode(pdf_bytes).decode('utf-8')

# Create event
event = {
    "requestContext": {"http": {"method": "POST"}},
    "headers": {"content-type": "application/json"},
    "body": json.dumps({
        "prompt": "Summarize this document",
        "email": "test@example.com",
        "session_id": "file-test-001",
        "file_content": base64_content,
        "file_type": "pdf"
    }),
    "isBase64Encoded": False
}

# Save
with open('events/real_pdf_test_true.json', 'w') as f:
    json.dump(event, f, indent=2)

print(f"Created test event with {len(pdf_bytes)} bytes PDF")
