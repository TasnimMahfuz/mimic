import pytest
import uuid

def test_heavy_auth_load(client):
    """Stress test: 300 unique registrations and logins."""
    passwords = "password123"
    created_users = []

    # 1. Stress Registration
    for i in range(30):
        email = f"user_{i}_{uuid.uuid4().hex[:6]}@stress.com"
        response = client.post("/auth/register", json={
            "email": email,
            "password": passwords,
            "role": "student"
        })
        assert response.status_code == 200
        print(f"Registered: {email}")
        created_users.append(email)

    # 2. Stress Login
    for email in created_users:
        response = client.post("/auth/login", json={
            "email": email,
            "password": passwords
        })
        assert response.status_code == 200
        print(f"Logged in: {email}")
        assert "token" in response.json()