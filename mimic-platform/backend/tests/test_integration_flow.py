import pytest

def test_full_teacher_workflow(client):
    # Step 1: Register
    email = "workflow_teacher@test.com"
    client.post("/auth/register", json={
        "email": email, "password": "pass", "role": "teacher"
    })

    # Step 2: Login to get token
    login_resp = client.post("/auth/login", json={"email": email, "password": "pass"})
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # Step 3: Sequential uploads (Stress testing the session)
    for i in range(5):
        files = {"file": (f"doc_{i}.txt", f"Content of doc {i}", "text/plain")}
        resp = client.post("/materials/upload", files=files, headers=headers)
        assert resp.status_code == 200
        assert "material_id" in resp.json()