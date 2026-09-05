from unittest.mock import MagicMock


def test_process_task_initializes_schema_before_opening_bandit_store(monkeypatch):
    from backend.queue import tasks

    calls: list[str] = []

    class SchemaStore:
        def __init__(self, _url):
            calls.append("event_store_created")

        def init_schema(self):
            calls.append("schema_initialized")

        def close(self):
            calls.append("event_store_closed")

    class BanditStore:
        def __init__(self, _url):
            calls.append("bandit_store_created")

        def load_policy(self, _domain):
            return None

        def close(self):
            calls.append("bandit_store_closed")

    orchestrator = MagicMock()
    event_store = MagicMock()
    monkeypatch.setattr("backend.storage.postgres_event_store.PostgresEventStore", SchemaStore)
    monkeypatch.setattr("backend.storage.postgres_bandit_store.PostgresBanditStore", BanditStore)
    monkeypatch.setattr("backend.wiring.build_orchestrator", lambda learning_core: (orchestrator, event_store))
    orchestrator.process_case.return_value = {"case_id": "case-1"}

    assert tasks.process_case_task.run("case-1", "subscription", {"decline_code": "51"}) == {"case_id": "case-1"}
    assert calls.index("schema_initialized") < calls.index("bandit_store_created")
