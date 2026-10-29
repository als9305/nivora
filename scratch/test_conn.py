import requests

try:
    res = requests.post(
        "http://localhost:8000/api/auth/deepfake",
        json={"user_id": "test", "frames_base64": ["data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEAYABgAAD/2wBDAP"]},
        timeout=5
    )
    print("Status:", res.status_code)
    print("Response:", res.text)
except Exception as e:
    print("Connection error:", e)
