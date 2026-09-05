import importlib
import ssl


def test_rediss_config_requires_certificate_validation(monkeypatch):
    monkeypatch.setenv("REVENIO_REDIS_URL", "rediss://default:secret@example.upstash.io:6379/0")

    import backend.queue.celery_app as celery_module

    celery_module = importlib.reload(celery_module)

    assert celery_module.celery_app.conf.broker_use_ssl["ssl_cert_reqs"] == ssl.CERT_REQUIRED
    assert celery_module.celery_app.conf.redis_backend_use_ssl["ssl_cert_reqs"] == ssl.CERT_REQUIRED
