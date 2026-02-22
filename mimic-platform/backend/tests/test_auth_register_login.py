import pytest

def test_register_teacher(client):
    r = client.post("/auth/register", json={
        "email": "teacher@test.com",
        "password": "1234",
        "role": "teacher"
    })
    data = r.json()
    assert r.status_code == 200
    assert "token" in data

def test_login_teacher(client):
    # first register
    client.post("/auth/register", json={
        "email": "teacher2@test.com",
        "password": "1234",
        "role": "teacher"
    })
    r = client.post("/auth/login", json={
        "email": "teacher2@test.com",
        "password": "1234"
    })
    data = r.json()
    assert r.status_code == 200
    assert "token" in data