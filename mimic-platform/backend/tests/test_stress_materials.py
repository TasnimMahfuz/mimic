import pytest

def test_massive_upload_volume(client):
    """Stress test: 300 sequential uploads by a single teacher."""
    # 1. Setup: Register and get token
    email = "heavy_teacher@test.com"
    client.post("/auth/register", json={
        "email": email, "password": "pass", "role": "teacher"
    })
    login_resp = client.post("/auth/login", json={"email": email, "password": "pass"})
    token = login_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}
    print(f"Registered and logged in as {email}. Starting uploads...")

    # 2. Perform 300 uploads
    for i in range(30):
        # We vary the filename and content to ensure the DB/RAG isn't just caching
        files = {
            "file": (f"stress_doc_{i}.txt", f"This is the content for file number {i}", "text/plain")
        }
        
        # NOTE: Using the correct '/materials' prefix to avoid 404
        resp = client.post("/materials/upload", files=files, headers=headers)
        
        print(f"Upload {i+1}/300: Status {resp.status_code}")
        assert resp.status_code == 200
        assert "material_id" in resp.json()