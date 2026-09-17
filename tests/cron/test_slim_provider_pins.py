"""A persisted cron pin cannot fall through to an unrelated configured provider."""
from types import SimpleNamespace
import pytest
from cron.scheduler import _resolve_job_runtime

@pytest.mark.parametrize('provider', ['openrouter', 'copilot-acp', 'moa'])
def test_retired_job_pin_rejected_before_fallback(provider):
    config = SimpleNamespace(model='test', cron_default_provider=None, model_cfg={},
        cfg={'fallback_providers': [{'provider': 'custom', 'model': 'test',
              'base_url': 'http://localhost:9999/v1', 'api_key': 'test-key'}]})
    with pytest.raises((RuntimeError, ValueError), match='Unsupported model provider'):
        _resolve_job_runtime({'provider': provider}, 'job-test', config)
