def test_register_teacher(client):

    r = client.post(
        "/auth/register",
        json={
            "email": "teacher@test.com",
            "password": "1234",
            "role": "teacher",
        },
    )

    print(r.json())

    assert r.status_code == 200
    assert "token" in r.json()