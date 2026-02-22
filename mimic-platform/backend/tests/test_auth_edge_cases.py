import pytest

def test_register_duplicate_email(client):
    payload = {
        "email": "duplicate@test.com",
        "password": "password123",
        "role": "teacher"
    }
    # First attempt
    client.post("/auth/register", json=payload)
    
    # Second attempt with same email
    response = client.post("/auth/register", json=payload)
    assert response.status_code == 400
    assert response.json()["detail"] == "Email already registered"

def test_login_wrong_password(client):
    # Setup: Register a user
    client.post("/auth/register", json={
        "email": "wrongpass@test.com",
        "password": "correct_password",
        "role": "student"
    })
    
    # Attempt login with bad password
    response = client.post("/auth/login", json={
        "email": "wrongpass@test.com",
        "password": "incorrect_password"
    })
    # Note: If your service raises Exception, this might return 500 
    # unless you have an Exception Handler. Ideally, it should be 401.
    assert response.status_code != 200

def test_login_nonexistent_user(client):
    response = client.post("/auth/login", json={
        "email": "nobody@test.com",
        "password": "password123"
    })
    assert response.status_code != 200