"""HTTP contracts for the supported model-account boundary."""
from unittest.mock import Mock

import pytest
from fastapi.testclient import TestClient
from hermes_cli.web_server import app, _SESSION_TOKEN
from hermes_cli import web_server_oauth as workers
from hermes_cli.web_routers import oauth

client = TestClient(app)
HEADERS = {"X-Hermes-Session-Token": _SESSION_TOKEN}


def test_account_catalog_only_offers_direct_copilot_and_nous(monkeypatch):
    monkeypatch.setattr(oauth, "_resolve_provider_status", lambda *a: {"logged_in": False})
    response = client.get("/api/providers/oauth", headers=HEADERS)
    assert response.status_code == 200
    rows = {r["id"]: r for r in response.json()["providers"]}
    assert set(rows) == {"copilot", "nous"}
    assert rows["copilot"]["flow"] == "device_code"
    assert rows["copilot"]["cli_command"] == "hermes auth add copilot"


@pytest.mark.parametrize("provider", ["openai-codex", "anthropic", "copilot-acp", "qwen-oauth", "minimax-oauth", "xai-oauth", "moa", "bedrock"])
def test_retired_oauth_rejected_before_start_or_status(provider, monkeypatch):
    start = Mock(side_effect=AssertionError("must not start authentication"))
    status = Mock(side_effect=AssertionError("must not resolve credentials"))
    monkeypatch.setattr(oauth, "_start_device_code_flow", start)
    monkeypatch.setattr(oauth, "_resolve_provider_status", status)
    assert client.post(f"/api/providers/oauth/{provider}/start", headers=HEADERS).status_code == 400
    assert client.delete(f"/api/providers/oauth/{provider}", headers=HEADERS).status_code == 400
    assert client.get(f"/api/providers/oauth/{provider}/poll/unused", headers=HEADERS).status_code == 400
    start.assert_not_called()
    status.assert_not_called()


def test_copilot_device_flow_persists_and_never_exposes_token(tmp_path, monkeypatch):
    from hermes_cli import copilot_auth, auth
    from agent.credential_pool import load_pool
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    calls = []
    def post(url, fields, timeout):
        calls.append(url)
        if url.endswith('/device/code'):
            return {"device_code": "private-device", "user_code": "ABCD-1234", "verification_uri": "https://github.com/login/device", "expires_in": 600, "interval": 1}
        return {"access_token": "ghu_test_device_token"}
    monkeypatch.setattr(copilot_auth, "_post_form", post)
    monkeypatch.setattr(oauth, "_start_poller", lambda *a, **k: None)
    monkeypatch.setattr(workers.time, "sleep", lambda *a: None)
    response = client.post('/api/providers/oauth/copilot/start', headers=HEADERS)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body['user_code'] == 'ABCD-1234'
    assert 'private-device' not in response.text
    sid = body['session_id']
    workers._copilot_poller(sid)
    poll = client.get(f'/api/providers/oauth/copilot/poll/{sid}', headers=HEADERS)
    assert poll.json()['status'] == 'approved', poll.text
    assert 'ghu_test_device_token' not in poll.text
    assert any(e.access_token == 'ghu_test_device_token' for e in load_pool('copilot').entries())
    assert len(calls) == 2
    monkeypatch.setattr(copilot_auth, 'resolve_copilot_token', lambda: ('', ''))
    disconnect = client.delete('/api/providers/oauth/copilot', headers=HEADERS)
    assert disconnect.status_code == 200, disconnect.text
    assert disconnect.json()['ok'] is True
    assert not auth.read_credential_pool('copilot')
    assert workers._copilot_status()['logged_in'] is False
    client.delete(f'/api/providers/oauth/sessions/{sid}', headers=HEADERS)


def test_nous_start_keeps_device_contract(monkeypatch):
    from hermes_cli import auth, anon_auth
    monkeypatch.setattr(anon_auth, 'guest_enabled', lambda: False)
    monkeypatch.setattr(oauth, '_start_poller', lambda *a, **k: None)
    monkeypatch.setattr(auth, '_request_device_code', lambda **k: {
        'device_code': 'private-nous-device', 'user_code': 'NOUS-1234',
        'verification_uri_complete': 'https://portal.nousresearch.com/device',
        'expires_in': 600, 'interval': 5})
    response = client.post('/api/providers/oauth/nous/start', headers=HEADERS)
    assert response.status_code == 200, response.text
    assert response.json()['flow'] == 'device_code'
    assert response.json()['user_code'] == 'NOUS-1234'
    assert 'private-nous-device' not in response.text
    client.delete('/api/providers/oauth/sessions/' + response.json()['session_id'], headers=HEADERS)


@pytest.mark.parametrize('error,status', [('access_denied', 'denied'), ('expired_token', 'expired')])
def test_copilot_terminal_states(error, status, monkeypatch):
    from hermes_cli import copilot_auth
    sid, sess = oauth._new_oauth_session('copilot', 'device_code')
    sess.update(device_code='private', interval=1, expires_at=workers.time.time() + 600)
    monkeypatch.setattr(workers.time, 'sleep', lambda *a: None)
    monkeypatch.setattr(copilot_auth, '_post_form', lambda *a: {'error': error})
    workers._copilot_poller(sid)
    assert client.get(f'/api/providers/oauth/copilot/poll/{sid}', headers=HEADERS).json()['status'] == status
    client.delete(f'/api/providers/oauth/sessions/{sid}', headers=HEADERS)


def test_recommended_default_rejects_retired_provider_before_catalog(monkeypatch):
    from hermes_cli import inventory
    catalog = Mock(side_effect=AssertionError('must not fetch model catalog'))
    monkeypatch.setattr(inventory, 'build_models_payload', catalog)
    response = client.get('/api/model/recommended-default?provider=openrouter', headers=HEADERS)
    assert response.status_code == 400
    catalog.assert_not_called()


def test_removed_routes_absent_from_openapi():
    paths = app.openapi()['paths']
    assert '/api/model/moa' not in paths
    assert '/api/providers/oauth/{provider_id}/submit' not in paths


@pytest.mark.parametrize('scope', ['main', 'auxiliary'])
def test_retired_model_assignment_rejects_before_pricing(scope, monkeypatch):
    from hermes_cli import model_selection_guards
    pricing = Mock(side_effect=AssertionError('no network before validation'))
    monkeypatch.setattr(model_selection_guards, 'combined_selection_warning', pricing)
    response = client.post('/api/model/set', headers=HEADERS, json={
        'scope': scope, 'task': 'vision', 'provider': 'openrouter', 'model': 'vendor/model'})
    assert response.status_code == 400, response.text
    pricing.assert_not_called()


@pytest.mark.parametrize('provider', ['nous', 'custom', 'custom:local'])
def test_supported_aux_assignment_writes_selected_provider(provider, tmp_path, monkeypatch):
    from hermes_cli.config import save_config, load_config
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    save_config({'custom_providers': [{'name': 'local', 'base_url': 'http://localhost:8000/v1'}]})
    response = client.post('/api/model/set', headers=HEADERS, json={
        'scope': 'auxiliary', 'task': 'vision', 'provider': provider, 'model': 'example',
        'base_url': 'http://localhost:8000/v1' if provider == 'custom' else '',
        'confirm_expensive_model': True})
    assert response.status_code == 200, response.text
    assert load_config()['auxiliary']['vision']['provider'] == provider


def test_copilot_cancel_during_token_request_prevents_persistence(tmp_path, monkeypatch):
    from hermes_cli import copilot_auth, auth
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    sid, sess = oauth._new_oauth_session('copilot', 'device_code')
    sess.update(device_code='private', interval=1, expires_at=workers.time.time() + 600)
    monkeypatch.setattr(workers.time, 'sleep', lambda *a: None)
    def post(*a):
        response = client.delete(f'/api/providers/oauth/sessions/{sid}', headers=HEADERS)
        assert response.status_code == 200
        return {'access_token': 'ghu_cancelled_token'}
    monkeypatch.setattr(copilot_auth, '_post_form', post)
    workers._copilot_poller(sid)
    assert not auth.read_credential_pool('copilot')
    assert sess['status'] == 'cancelled'


def test_copilot_failures_do_not_leak_provider_exception(monkeypatch, caplog):
    from hermes_cli import copilot_auth
    sid, sess = oauth._new_oauth_session('copilot', 'device_code')
    sess.update(device_code='private', interval=1, expires_at=workers.time.time() + 600)
    monkeypatch.setattr(workers.time, 'sleep', lambda *a: None)
    monkeypatch.setattr(copilot_auth, '_post_form', Mock(side_effect=ValueError('secret-response-body')))
    workers._copilot_poller(sid)
    response = client.get(f'/api/providers/oauth/copilot/poll/{sid}', headers=HEADERS)
    assert response.json()['status'] == 'error'
    assert 'secret-response-body' not in response.text
    assert 'secret-response-body' not in caplog.text
    client.delete(f'/api/providers/oauth/sessions/{sid}', headers=HEADERS)
