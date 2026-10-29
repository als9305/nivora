import httpx

def main():
    base_url = "http://localhost:8000"
    
    # 1. Health check
    r1 = httpx.get(f"{base_url}/health")
    print("Health Status:", r1.status_code, r1.json())
    
    # 2. Registration & Face Enrollment test
    reg_payload = {
        "name": "Security Analyst",
        "email": "analyst@deepguard.sec",
        "password": "password123"
    }
    r2 = httpx.post(f"{base_url}/api/auth/register", json=reg_payload)
    print("Register Status:", r2.status_code, r2.json())

    r2_enroll = httpx.post(f"{base_url}/api/auth/enroll-face", json={"email": "analyst@deepguard.sec"})
    print("Enroll-Face Status:", r2_enroll.status_code, r2_enroll.json())
    
    # 3. Login test
    login_payload = {
        "email": "analyst@deepguard.sec",
        "password": "password123"
    }
    r3 = httpx.post(f"{base_url}/api/auth/login", json=login_payload)
    print("Login Status:", r3.status_code, r3.json())

    # 4. Simulation Attack Tests (Deepfake & Genuine)
    r4 = httpx.post(f"{base_url}/api/auth/simulate?simulation_type=DEEPFAKE_ATTACK")
    print("Deepfake Simulation Status:", r4.status_code, r4.json()["decision"], r4.json()["risk_score"])

    r5 = httpx.post(f"{base_url}/api/auth/simulate?simulation_type=GENUINE_USER")
    print("Genuine User Simulation Status:", r5.status_code, r5.json()["decision"], r5.json()["risk_score"])

    # 5. Dashboard stats test
    r6 = httpx.get(f"{base_url}/api/dashboard/stats")
    print("Dashboard Stats:", r6.status_code, r6.json())

if __name__ == "__main__":
    main()
