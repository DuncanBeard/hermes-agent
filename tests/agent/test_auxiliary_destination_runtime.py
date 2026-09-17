"""Real SDK requests exercise the shared resolver, never replacement credentials."""
import json

import httpx
import pytest
import yaml


@pytest.fixture
def wire(tmp_path, monkeypatch):
    from openai import OpenAI, AsyncOpenAI
    import agent.auxiliary_client as aux
    monkeypatch.setenv('HERMES_HOME', str(tmp_path))
    for name in ('CUSTOM_BASE_URL', 'CUSTOM_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_API_KEY'):
        monkeypatch.delenv(name, raising=False)
    aux.clear_runtime_main()
    aux.shutdown_cached_clients()
    requests = []
    def respond(request):
        requests.append(request)
        if request.url.path.endswith('/responses'):
            response = {'id': 'resp_test', 'object': 'response',
                'created_at': 0, 'status': 'completed', 'model': 'test-model',
                'output': [{'type': 'message', 'id': 'msg_test', 'role': 'assistant',
                    'content': [{'type': 'output_text', 'text': 'ok', 'annotations': []}]}]}
            return httpx.Response(200, headers={'content-type': 'text/event-stream'},
                text='data: ' + json.dumps({'type': 'response.output_item.done', 'output_index': 0,
                    'item': response['output'][0]}) + '\n\ndata: ' +
                    json.dumps({'type': 'response.completed', 'response': response}) + '\n\n')
        return httpx.Response(200, json={'id': 'test', 'object': 'chat.completion',
            'created': 0, 'model': 'test-model', 'choices': [{'index': 0,
            'message': {'role': 'assistant', 'content': 'ok'}, 'finish_reason': 'stop'}]})
    def client(**kwargs):
        kwargs['http_client'] = httpx.Client(transport=httpx.MockTransport(respond))
        return OpenAI(**kwargs)
    monkeypatch.setattr(aux, 'OpenAI', client)
    def async_client(**kwargs):
        kwargs['http_client'] = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        return AsyncOpenAI(**kwargs)
    monkeypatch.setattr('openai.AsyncOpenAI', async_client)
    def configure(config):
        (tmp_path / 'config.yaml').write_text(yaml.safe_dump(config), encoding='utf-8')
    yield aux, configure, requests
    aux.clear_runtime_main()
    aux.shutdown_cached_clients()


def send(client):
    try:
        result = client.chat.completions.create(model='test-model', messages=[{'role': 'user', 'content': 'hello'}])
        assert result.choices[0].message.content == 'ok'
    finally:
        client.close()


@pytest.mark.parametrize('provider', ['custom', 'custom:original'])
@pytest.mark.parametrize('target', [
    'http://original.example.test/v2',
    'http://original.example.test/v1',
    'https://original.example.test:8443/v1',
    'https://original.example.test/v2',
    'https://original.example.test/V1',
    'https://original.example.test.attacker.test/v1',
    'https://different.example.test/v1',
])
@pytest.mark.parametrize('credential', ['api_key', 'key_env', 'key_cmd'])
def test_override_never_reuses_original_credentials(wire, monkeypatch, provider, target, credential):
    aux, configure, requests = wire
    entry = {'base_url': 'https://original.example.test/v1',
        'extra_headers': {'X-Original-Auth': 'original-header'},
        'extra_body': {'original_auth': 'original-body'}}
    commands = []
    def token_source(command, label):
        commands.append(command)
        return lambda: 'original-command-key'
    monkeypatch.setattr('agent.command_token_source.build_command_token_provider', token_source)
    if credential == 'api_key':
        entry[credential] = 'original-key'
    elif credential == 'key_env':
        entry[credential] = 'SYNTHETIC_ORIGINAL_KEY'
        monkeypatch.setenv('SYNTHETIC_ORIGINAL_KEY', 'original-env-key')
    else:
        entry[credential] = 'synthetic-command'
    configure({'model': {'provider': provider, 'default': 'test-model', **entry},
               'providers': {'original': entry}})
    client, _ = aux.resolve_provider_client(provider, 'test-model', explicit_base_url=target)
    send(client)
    assert len(requests) == 1
    request = requests[0]
    assert str(request.url) == target + '/chat/completions'
    assert request.headers['Authorization'] == 'Bearer no-key-required'
    assert 'X-Original-Auth' not in request.headers
    assert 'original_auth' not in json.loads(request.content)
    assert commands == []


@pytest.mark.parametrize('provider', ['custom', 'custom:original'])
@pytest.mark.parametrize('route', ['same', 'named', 'explicit_key', 'environment'])
def test_destination_owned_runtime_is_complete(wire, monkeypatch, provider, route):
    aux, configure, requests = wire
    original = 'https://original.example.test/v1'
    target = original if route == 'same' else 'https://target.example.test/v1'
    target_entry = {'base_url': target, 'key_cmd': 'target-command',
        'extra_headers': {'X-Target-Auth': 'target-header'},
        'extra_body': {'target_auth': 'target-body'}, 'capabilities': {'vision': True}}
    original_entry = {'base_url': original, 'api_key': 'original-key'}
    configure({'model': {'provider': 'custom', 'default': 'test-model', **original_entry},
        'providers': {'original': target_entry if route == 'same' else original_entry,
                      **({'target': target_entry} if route != 'same' else {})}})
    minted = []
    def source(command, label):
        assert command == 'target-command'
        def token():
            minted.append(command)
            return 'target-command-key'
        return token
    monkeypatch.setattr('agent.command_token_source.build_command_token_provider', source)
    kwargs = {'explicit_base_url': 'https://ORIGINAL.example.test:443/v1/' if route == 'same' else target}
    if route == 'explicit_key':
        kwargs['explicit_api_key'] = 'explicit-target-key'
    if route == 'environment':
        monkeypatch.setenv('CUSTOM_BASE_URL', target)
        kwargs = {}
        provider = 'custom'
    client, _ = aux.resolve_provider_client(provider, 'test-model', **kwargs)
    assert minted == []
    send(client)
    request = requests[0]
    assert str(request.url) == target + '/chat/completions'
    assert request.headers['Authorization'] == 'Bearer ' + ('explicit-target-key' if route == 'explicit_key' else 'target-command-key')
    assert request.headers['X-Target-Auth'] == 'target-header'
    assert json.loads(request.content)['target_auth'] == 'target-body'
    assert minted == ([] if route == 'explicit_key' else ['target-command'])


@pytest.mark.parametrize('route', ['direct', 'auto', 'cached'])
def test_main_runtime_override_keeps_its_destination_and_callable(wire, route):
    aux, configure, requests = wire
    configure({'model': {'provider': 'custom:original', 'default': 'test-model'},
        'providers': {'original': {'base_url': 'https://original.example.test/v1', 'api_key': 'original-key'}}})
    minted = []
    def token():
        minted.append(True)
        return 'runtime-target-key'
    runtime = {'provider': 'custom:original', 'model': 'test-model',
        'base_url': 'https://target.example.test/v1', 'api_key': token, 'api_mode': 'chat_completions'}
    if route == 'direct':
        client, _ = aux.resolve_provider_client('custom', 'test-model', main_runtime=runtime)
    elif route == 'auto':
        client, _ = aux.resolve_provider_client('auto', 'test-model', main_runtime=runtime)
    else:
        client, _ = aux._get_cached_client('auto', 'test-model', main_runtime=runtime)
        cached, _ = aux._get_cached_client('auto', 'test-model', main_runtime=runtime)
        assert cached is client
    assert minted == []
    send(client)
    assert str(requests[0].url) == 'https://target.example.test/v1/chat/completions'
    assert requests[0].headers['Authorization'] == 'Bearer runtime-target-key'
    assert minted == [True]


@pytest.mark.parametrize('provider', ['custom', 'custom:original'])
def test_userinfo_endpoint_rejected_before_request(wire, provider):
    aux, configure, requests = wire
    configure({'model': {'provider': 'custom', 'base_url': 'https://original.example.test/v1', 'api_key': 'original-key'},
        'providers': {'original': {'base_url': 'https://original.example.test/v1', 'api_key': 'original-key'}}})
    with pytest.raises(ValueError):
        aux.resolve_provider_client(provider, 'test-model', explicit_base_url='https://original.example.test@attacker.test/v1')
    assert requests == []


@pytest.mark.parametrize('route', ['custom', 'auto'])
@pytest.mark.parametrize('mode', ['chat_completions', 'codex_responses'])
@pytest.mark.parametrize('async_mode', [False, True])
def test_cached_runtime_snapshot_preserves_protocol_and_extras(wire, route, mode, async_mode):
    import asyncio
    aux, configure, requests = wire
    configure({'model': {'provider': 'openrouter', 'default': 'stale'},
        'providers': {'original': {'base_url': 'https://original.example.test/v1', 'api_key': 'original-key'}}})
    runtime = {'provider': 'custom:original', 'model': 'test-model',
        'base_url': 'https://target.example.test/v1', 'api_key': lambda: 'runtime-key',
        'api_mode': mode, 'extra_headers': {'X-Runtime': 'runtime-header'},
        'request_overrides': {'extra_body': {'runtime_auth': 'runtime-body'}},
        'capabilities': {'vision': True}}
    async def exercise():
        client, model = aux._get_cached_client(route, main_runtime=runtime, async_mode=async_mode)
        assert model == 'test-model'
        assert client._hermes_custom_runtime['capabilities'] == {'vision': True}
        result = client.chat.completions.create(model=model, messages=[{'role': 'user', 'content': 'hello'}])
        if async_mode:
            result = await result
        assert result.choices[0].message.content == 'ok'
        runtime['extra_headers']['X-Runtime'] = 'changed-header'
        changed, _ = aux._get_cached_client(route, main_runtime=runtime, async_mode=async_mode)
        assert changed is not client
    asyncio.run(exercise())
    assert str(requests[0].url) == 'https://target.example.test/v1/' + ('responses' if mode == 'codex_responses' else 'chat/completions')
    assert requests[0].headers['Authorization'] == 'Bearer runtime-key'
    assert requests[0].headers['X-Runtime'] == 'runtime-header'
    assert json.loads(requests[0].content)['runtime_auth'] == 'runtime-body'


@pytest.mark.parametrize('provider', ['custom', 'custom:original'])
def test_cached_custom_credentials_resolved_only_for_destination(wire, monkeypatch, provider):
    from types import SimpleNamespace
    aux, configure, requests = wire
    configure({'model': {'provider': 'custom', 'base_url': 'https://original.example.test/v1'},
        'providers': {'original': {'base_url': 'https://original.example.test/v1', 'api_key': 'original-key'}}})
    monkeypatch.setattr(aux, '_peek_pool_entry', lambda name: SimpleNamespace(runtime_api_key='unbound-pool-key'))
    client, _ = aux._get_cached_client(provider, 'test-model', base_url='https://target.example.test/v1')
    send(client)
    assert requests[0].headers['Authorization'] == 'Bearer no-key-required'
