from app import app

def test_index_get():
    client = app.test_client()
    response = client.get("/")
    assert response.status_code == 200
