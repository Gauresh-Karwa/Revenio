from backend.api.runtime import DemoRuntime


def test_overdue_invoice_runs_through_human_gate_then_hinglish_voice_delivery():
    runtime = DemoRuntime()

    runtime.simulate("overdue_invoice")
    review = runtime.reviews()[0]
    assert review["domain_type"] == "b2b_receivables"
    assert review["status"] == "HUMAN_REVIEW"

    detail = runtime.resolve_review(review["case_id"], approved=True)
    executions = [event for event in detail["events"] if event["event_type"] == "ExecutionResult"]

    assert executions[-1]["payload"]["details"]["channel"] == "voice"
    assert "Hinglish call response:" in executions[-1]["payload"]["details"]["response"]
