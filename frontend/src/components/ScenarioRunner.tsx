import type { Scenario } from "../types";

type Props = {
  scenario: Scenario;
  busy: boolean;
  onScenarioChange: (scenario: Scenario) => void;
  onRun: () => void;
};

export function ScenarioRunner({ scenario, busy, onScenarioChange, onRun }: Props) {
  return <section className="simulator">
    <div>
      <p className="eyebrow">CASE EXECUTION</p>
      <strong>Create a fresh recovery incident</strong>
      <small>Every request generates new data and runs the existing decision pipeline.</small>
    </div>
    <select value={scenario} onChange={(event) => onScenarioChange(event.target.value as Scenario)}>
      <option value="random">Mixed merchant burst (5 cases)</option>
      <option value="payment_failure">Payment failure</option>
      <option value="checkout_abandonment">Checkout abandonment</option>
      <option value="overdue_invoice">Invoice: email → SMS → Hinglish voice</option>
      <option value="mandate_failure">UPI AutoPay failure</option>
    </select>
    <button className="primary" disabled={busy} onClick={onRun}>{busy ? "Processing…" : "Execute case"}</button>
  </section>;
}
