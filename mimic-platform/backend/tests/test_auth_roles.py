import pytest

def test_register_student(client):
    r = client.post("/auth/register", json={
        "email": "student@test.com",
        "password": "1234",
        "role": "student"
    })
    data = r.json()
    assert r.status_code == 200
    assert "token" in data

def test_invalid_role(client):
    r = client.post("/auth/register", json={
        "email": "hacker@test.com",
        "password": "1234",
        "role": "admin"
    })
    data = r.json()
    assert r.status_code == 400
    assert "detail" in data