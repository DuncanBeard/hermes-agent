"""Tests for the provider module registry and profiles."""

from providers import get_provider_profile


class TestRegistry:
    def test_discovery_populates_registry(self):
        p = get_provider_profile("custom")
        assert p is not None
        assert p.name == "custom"


class TestNousProfile:
    def test_tags(self):
        from agent.portal_tags import nous_portal_tags
        p = get_provider_profile("nous")
        body = p.build_extra_body()
        assert body["tags"] == nous_portal_tags()

    def test_sticky_session_id_normalizes_cron_timestamp(self):
        """Cron re-fires of the same job keep the same sticky routing key."""
        p = get_provider_profile("nous")
        first = p.build_extra_body(session_id="cron_job42_20260801_090000")
        second = p.build_extra_body(session_id="cron_job42_20260802_090000")
        assert first["session_id"] == "cron_job42"
        assert first["session_id"] == second["session_id"]

    def test_extra_body_ignores_provider_preferences(self):
        """Nous Portal rejects caller-supplied provider routing prefs (HTTP 400)."""
        p = get_provider_profile("nous")
        body = p.build_extra_body(
            provider_preferences={"allow": ["anthropic"], "sort": "price"}
        )
        assert "provider" not in body
        assert "tags" in body


    def test_auth_type(self):
        p = get_provider_profile("nous")
        assert p.auth_type == "oauth_device_code"
