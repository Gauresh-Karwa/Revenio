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


def test_high_value_mandate_requires_review_then_resumes_after_approval():
    runtime = DemoRuntime()

    detail = runtime.simulate_custom(
        domain_type="mandate_retry",
        customer_name="Nila Kapoor",
        customer_id="merchant-customer-test",
        customer_email=None,
        customer_phone=None,
        amount=15_001,
        failure_code="U02",
        response="no_response",
    )
    assert any(event["event_type"] == "PendingHumanReview" for event in detail["events"])

    approved = runtime.resolve_review(detail["state"]["case_id"], approved=True)
    decisions = [event for event in approved["events"] if event["event_type"] == "Decision"]
    assert decisions[-1]["payload"]["action_params"]["channel"] == "push_notification"
    assert decisions[-1]["payload"]["requires_human_review"] is False


def test_nach_return_is_preserved_as_rail_specific_audit_evidence():
    runtime = DemoRuntime()

    detail = runtime.simulate_custom(
        domain_type="mandate_retry",
        customer_name="Nila Kapoor",
        customer_id="merchant-customer-test",
        customer_email=None,
        customer_phone=None,
        amount=3_500,
        failure_code="1",
        response="no_response",
        mandate_rail="nach",
    )

    diagnosis = next(event["payload"] for event in detail["events"] if event["event_type"] == "Diagnosis")
    assert diagnosis["raw_signal"]["rail"] == "nach"
    assert diagnosis["raw_signal"]["return_code"] == "1"
    assert any(event["event_type"] == "PendingHumanReview" for event in detail["events"])


def test_dashboard_counts_only_verified_razorpay_payment_revenue():
    runtime = DemoRuntime()
    detail = runtime.simulate_custom(
        domain_type="subscription",
        customer_name="Nila Kapoor",
        customer_id="merchant-customer-test",
        customer_email=None,
        customer_phone=None,
        amount=1_499,
        failure_code="51",
        response="recovered",
    )
    case_id = detail["state"]["case_id"]

    assert runtime.list_cases()[0]["status"] == "ACTIVE"
    assert runtime.dashboard()["money_recovered"] == 0
    assert runtime.dashboard()["recovery_rate"] == 0

    runtime.record_razorpay_payment_event(
        case_id, "payment.captured", {"id": "pay_test", "status": "captured", "amount": 149_900}
    )
    assert runtime.list_cases()[0]["status"] == "RECOVERED"
    assert runtime.dashboard()["money_recovered"] == 1_499
    assert runtime.dashboard()["recovery_rate"] == 100
