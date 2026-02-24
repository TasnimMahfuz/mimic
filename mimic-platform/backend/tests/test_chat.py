def test_chat_endpoint(client):

    r = client.post(
        "/query",
        json={"query": "What is machine learning?"}
    )

    assert r.status_code == 200
    assert "response" in r.json()