"""Retired MoA routes must not mutate existing configuration."""
from fastapi.testclient import TestClient
from hermes_cli.web_server import app, _SESSION_TOKEN


def test_retired_moa_write_preserves_existing_config(tmp_path, monkeypatch):
    from hermes_cli.config import get_config_path
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    path = get_config_path()
    original = 'moa:\n  save_traces: true\n  trace_dir: /custom/traces\n'
    path.write_text(original, encoding='utf-8')
    response = TestClient(app).put('/api/model/moa', json={'enabled': False},
                                  headers={'X-Hermes-Session-Token': _SESSION_TOKEN})
    # A built dashboard mounts StaticFiles at /, whose PUT fallback is 405.
    assert response.status_code in {404, 405}
    assert path.read_text(encoding='utf-8') == original
