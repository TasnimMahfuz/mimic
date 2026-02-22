import pytest

def test_student_cannot_upload_material(client):
    # 1. Register as a student
    reg_resp = client.post("/auth/register", json={
        "email": "student_hacker@test.com",
        "password": "1234",
        "role": "student"
    })
    token = reg_resp.json()["token"]

    # 2. Try to upload a file using the student token
    files = {"file": ("test.txt", "some content", "text/plain")}
    headers = {"Authorization": f"Bearer {token}"}
    
    response = client.post("/materials/upload", files=files, headers=headers)
    
    # This should fail with 403 Forbidden because of @require_teacher
    assert response.status_code == 403
    assert response.json()["detail"] == "Teacher only"

def test_upload_empty_file(client):
    # 1. Register as teacher
    reg_resp = client.post("/auth/register", json={
        "email": "teacher_real@test.com",
        "password": "1234",
        "role": "teacher"
    })
    token = reg_resp.json()["token"]
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Upload an empty file
    files = {"file": ("empty.txt", "", "text/plain")}
    response = client.post("/materials/upload", files=files, headers=headers)
    
    # Verify how your rag_service handles empty content
    assert response.status_code == 200