import pytest
from impsy.web_interface import app, get_routes

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_index_route(client):
    response = client.get('/')
    assert response.status_code == 200
    assert b'IMPSY Web Interface' in response.data

def test_routes_exist():
    routes = get_routes()
    assert len(routes) > 0
    for route in routes:
        assert 'endpoint' in route
        assert 'route' in route

def test_logs_route(client):
    response = client.get('/logs')
    assert response.status_code == 200

def test_config_route(client):
    response = client.get('/config')
    assert response.status_code == 200

# def test_download_route(client, log_file):
#     # This test assumes there's a file named 'test.log' in your LOGS_DIR
#     response = client.get('/download/test.log')
#     assert response.status_code == 200
#     assert response.headers['Content-Disposition'].startswith('attachment')
