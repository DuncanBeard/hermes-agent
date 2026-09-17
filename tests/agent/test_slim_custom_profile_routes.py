"""Real custom resolver/client integration under alternating served-profile scopes."""
import os
from pathlib import Path
import httpx
import pytest


def test_custom_requests_keep_profile_endpoint_and_key_a_b_a(tmp_path, monkeypatch):
    import tui_gateway.server as server
    import agent.auxiliary_client as aux
    from openai import OpenAI
    from agent.secret_scope import is_multiplex_active, set_multiplex_active

    homes = {}
    for label in ('a', 'b'):
        home = tmp_path / label
        home.mkdir()
        (home / '.env').write_text(f'LAB_KEY=test-key-{label}\n', encoding='utf-8')
        (home / 'config.yaml').write_text(
            'model:\n  provider: custom:lab\n  default: test-model\n'
            f'providers:\n  lab:\n    base_url: https://{label}.example.test/v1\n'
            '    key_env: LAB_KEY\n    default_model: test-model\n', encoding='utf-8')
        homes[label] = home
    requests = []
    def respond(request):
        requests.append((request.url.host, request.headers['authorization']))
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion',
            'created': 0, 'model': 'test-model',
            'choices': [{'index': 0, 'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})
    def client(**kwargs):
        kwargs['http_client'] = httpx.Client(transport=httpx.MockTransport(respond))
        return OpenAI(**kwargs)
    monkeypatch.setattr(aux, 'OpenAI', client)
    monkeypatch.setenv('LAB_KEY', 'launch-key-must-not-leak')
    was_active = is_multiplex_active()
    set_multiplex_active(True)
    try:
        for label in ('a', 'b', 'a'):
            with server._session_profile_runtime_scope({'profile_home': homes[label]}):
                resolved, model = aux.resolve_provider_client('custom:lab')
                try:
                    result = resolved.chat.completions.create(model=model,
                        messages=[{'role': 'user', 'content': 'hello'}])
                    assert result.choices[0].message.content == 'ok'
                finally:
                    resolved.close()
        assert requests == [('a.example.test', 'Bearer test-key-a'),
                            ('b.example.test', 'Bearer test-key-b'),
                            ('a.example.test', 'Bearer test-key-a')]
        assert os.environ['LAB_KEY'] == 'launch-key-must-not-leak'
    finally:
        set_multiplex_active(was_active)


@pytest.mark.parametrize('model, mode', [('claude-sonnet-4', 'chat_completions'),
                                        ('gpt-5.4', 'codex_responses')])
def test_copilot_wire_follows_model_not_stale_api_mode(model, mode):
    from types import SimpleNamespace
    from agent.agent_init import _resolve_api_mode
    agent = SimpleNamespace(provider='copilot', model=model)
    _resolve_api_mode(agent, 'anthropic_messages', 'copilot', 'https://api.githubcopilot.com')
    assert agent.api_mode == mode
