"""Generic auth error classification, independent of retired account transports."""

def test_is_rate_limited_auth_error_distinguishes_credential_errors():
    """Missing/expired credentials must NOT be treated as rate-limit errors."""
    from hermes_cli.auth import AuthError, CODEX_RATE_LIMITED_CODE, is_rate_limited_auth_error

    rate_limited = AuthError(
        "quota", provider="nous", code=CODEX_RATE_LIMITED_CODE, relogin_required=False
    )
    missing_creds = AuthError(
        "No credentials stored.",
        provider="nous",
        code="auth_missing",
        relogin_required=True,
    )
    assert is_rate_limited_auth_error(rate_limited) is True
    assert is_rate_limited_auth_error(missing_creds) is False
    assert is_rate_limited_auth_error(ValueError("nope")) is False
