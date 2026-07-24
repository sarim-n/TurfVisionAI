"""
Unit tests for configuration loading.
"""

from shared.config import Settings, get_settings


def test_settings_default_values():
    settings = Settings()
    assert settings.APP_NAME == "TurfVision AI"
    assert settings.ENVIRONMENT in ["development", "testing", "production"]
    assert settings.DEFAULT_TARGET_FPS == 30
    assert settings.CONFIDENCE_THRESHOLD == 0.35


def test_get_settings_caching():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2
