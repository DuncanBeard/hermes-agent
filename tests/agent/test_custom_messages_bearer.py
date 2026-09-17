"""Exercise the optional Messages SDK with rotating custom bearer credentials."""
import httpx
import pytest

def test_custom_messages_rotates_bearer_without_hosted_identity_sdk(monkeypatch):
    pytest.importorskip("anthropic", reason="optional custom Messages SDK is not installed")
    from agent import anthropic_adapter as adapter
    from agent import bearer_auth
    real_build = bearer_auth.build_bearer_http_client
    sent = []
    tokens = iter(["custom-one", "custom-two"])
    def respond(request):
        sent.append(dict(request.headers))
        return httpx.Response(200, json={"id": "msg_test", "type": "message", "role": "assistant",
            "content": [{"type": "text", "text": "ok"}], "model": "custom-model", "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 1, "output_tokens": 1}})
    monkeypatch.setattr(bearer_auth, "build_bearer_http_client", lambda provider, **kw:
        real_build(provider, transport=httpx.MockTransport(respond), **kw))
    client = adapter.build_anthropic_client(lambda: next(tokens), "https://custom.example.test/v1")
    try:
        for _ in range(2):
            result = client.messages.create(model="custom-model", max_tokens=10,
                messages=[{"role": "user", "content": "hi"}])
            assert result.content[0].text == "ok"
        assert [h["authorization"] for h in sent] == ["Bearer custom-one", "Bearer custom-two"]
        assert all("x-api-key" not in h and "api-key" not in h for h in sent)
    finally:
        client.close()
