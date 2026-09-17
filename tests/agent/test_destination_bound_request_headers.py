"""Credential isolation asserted at the real SDK HTTP request boundary."""
import httpx
import pytest
import yaml


@pytest.mark.parametrize('surface', ['main', 'aux'])
@pytest.mark.parametrize('route', ['explicit', 'environment', 'same', 'named'])
def test_config_headers_bound_to_destination(tmp_path, monkeypatch, surface, route):
    from openai import OpenAI
    import agent.auxiliary_client as aux
    import agent.process_bootstrap as bootstrap
    from run_agent import AIAgent

    original = 'https://original.example.test/v1'
    target = 'https://target.example.test/v1'
    config = {'model': {'provider': 'custom', 'default': 'test-model',
        'base_url': original, 'api_key': 'original-key',
        'default_headers': {'X-Original-Auth': 'original-secret', 'User-Agent': 'waf-agent'},
        'extra_headers': {'X-Original-Alias': 'alias-secret'}},
        'providers': {'target': {'base_url': target, 'api_key': 'target-key',
            'extra_headers': {'X-Target-Auth': 'target-secret'}}}}
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    (tmp_path / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    requests = []
    def respond(request):
        requests.append(request)
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion',
            'created': 0, 'model': 'test-model', 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})
    def client(**kwargs):
        kwargs['http_client'] = httpx.Client(transport=httpx.MockTransport(respond))
        return OpenAI(**kwargs)
    monkeypatch.setattr(aux, 'OpenAI', client)
    monkeypatch.setattr(bootstrap, 'OpenAI', client)
    destination = 'https://ORIGINAL.example.test:443/v1/' if route == 'same' else target
    provider = 'custom:target' if route == 'named' else 'custom'
    if route == 'environment':
        monkeypatch.setenv('CUSTOM_BASE_URL', target)
        monkeypatch.setenv('CUSTOM_API_KEY', 'target-key')
    if surface == 'main':
        kwargs = {} if route in ('environment', 'named') else {'base_url': destination, 'api_key': 'target-key'}
        agent = AIAgent(provider=provider, model='test-model', quiet_mode=True,
            skip_context_files=True, skip_memory=True, **kwargs)
        resolved = agent.client
    else:
        kwargs = {} if route in ('environment', 'named') else {'base_url': destination, 'api_key': 'target-key'}
        resolved, _ = aux.resolve_provider_client(provider, 'test-model', **{
            'explicit_' + key: value for key, value in kwargs.items()})
    try:
        result = resolved.chat.completions.create(model='test-model', messages=[{'role': 'user', 'content': 'hello'}])
        assert result.choices[0].message.content == 'ok'
    finally:
        resolved.close()
    assert len(requests) == 1
    headers = requests[0].headers
    assert requests[0].url.host == ('original.example.test' if route == 'same' else 'target.example.test')
    if route == 'same':
        assert headers['X-Original-Auth'] == 'original-secret'
        assert headers['X-Original-Alias'] == 'alias-secret'
        assert headers['User-Agent'] == 'waf-agent'
    else:
        assert 'X-Original-Auth' not in headers
        assert 'X-Original-Alias' not in headers
        assert headers['X-Target-Auth'] == 'target-secret'


@pytest.mark.parametrize('destination', ['', 'https://other.example.test/v1', 'https://original.example.test/V1'])
def test_unknown_or_different_destination_preserves_only_runtime_headers(monkeypatch, destination):
    from agent.auxiliary_client import _apply_user_default_headers
    config = {'model': {'base_url': 'https://original.example.test/v1',
        'default_headers': {'X-Source-Auth': 'source'}}}
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: config)
    monkeypatch.setattr('hermes_cli.runtime_provider.resolve_runtime_provider',
        lambda **kw: pytest.fail('header ownership must not resolve credentials'))
    assert _apply_user_default_headers({'X-Runtime-Auth': 'target'}, destination) == {'X-Runtime-Auth': 'target'}


@pytest.mark.parametrize('provider,destination', [
    ('nous', 'https://inference-api.nousresearch.com/v1'),
    ('copilot', 'https://api.githubcopilot.com'),
    ('openai-codex', 'https://chatgpt.com/backend-api/codex'),
])
def test_static_account_endpoint_owns_config_headers(monkeypatch, provider, destination):
    from agent.auxiliary_client import _apply_user_default_headers
    monkeypatch.setattr('hermes_cli.config.load_config', lambda: {'model': {
        'provider': provider, 'default_headers': {'X-Waf': 'allowed'}}})
    assert _apply_user_default_headers({'X-Route': 'retained'}, destination, provider) == {
        'X-Route': 'retained', 'X-Waf': 'allowed'}
    assert _apply_user_default_headers({'X-Route': 'retained'}, 'https://other.example.test', provider) == {
        'X-Route': 'retained'}
