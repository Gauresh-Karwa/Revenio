import { useMemo, useState } from "react";
import type { CustomSimulation, RecoveryDomain } from "../types";

type Props = { busy: boolean; onSubmit: (request: CustomSimulation) => void };
type Signal = { value: string; label: string; effect: string };

const domains: { value: RecoveryDomain; code: string; title: string; description: string }[] = [
  { value: "subscription", code: "CARD", title: "Subscription payment", description: "Issuer decline, guarded retry" },
  { value: "checkout_abandonment", code: "CHECKOUT", title: "Checkout recovery", description: "Consented return to checkout" },
  { value: "b2b_receivables", code: "INVOICE", title: "Invoice collection", description: "Invoice follow-up and review" },
  { value: "mandate_retry", code: "MANDATE", title: "UPI AutoPay / NACH", description: "Rail-aware return handling" },
];

const responses: Record<RecoveryDomain, { value: CustomSimulation["response"]; label: string }[]> = {
  subscription: [{ value: "recovered", label: "Retry succeeds" }, { value: "lost", label: "Retry is unsuccessful" }, { value: "hardship", label: "Customer asks for hardship support" }],
  checkout_abandonment: [{ value: "recovered", label: "Customer completes payment" }, { value: "lost", label: "Customer does not return" }, { value: "no_response", label: "No response to the message" }],
  b2b_receivables: [{ value: "paid", label: "Customer pays after contact" }, { value: "promise", label: "Customer promises a date" }, { value: "no_response", label: "No response - route to review" }, { value: "needs_human", label: "Customer asks for a specialist" }],
  mandate_retry: [{ value: "recovered", label: "Mandate succeeds after permitted action" }, { value: "lost", label: "Mandate is not completed" }, { value: "no_response", label: "Customer confirmation is pending" }],
};

const subscriptionSignals: Signal[] = [
  { value: "51", label: "51 — Insufficient funds", effect: "A bounded retry may be assessed." },
  { value: "05", label: "05 — Do not honour", effect: "The issuer response is assessed before a retry." },
  { value: "91", label: "91 — Issuer unavailable", effect: "A delayed retry may be assessed." },
  { value: "96", label: "96 — System malfunction", effect: "A technical retry may be assessed." },
  { value: "65", label: "65 — Activity limit exceeded", effect: "The retry policy considers issuer limits." },
  { value: "61", label: "61 — Withdrawal limit exceeded", effect: "The retry policy considers the limit signal." },
  { value: "43", label: "43 — Stolen card", effect: "Compliance stop. No automated retry." },
  { value: "04", label: "04 — Pickup card", effect: "Compliance stop. No automated retry." },
  { value: "07", label: "07 — Pickup card (special)", effect: "Compliance stop. No automated retry." },
  { value: "12", label: "12 — Invalid transaction", effect: "Compliance stop. No automated retry." },
  { value: "14", label: "14 — Invalid card number", effect: "Compliance stop. No automated retry." },
  { value: "15", label: "15 — Invalid issuer", effect: "Compliance stop. No automated retry." },
  { value: "41", label: "41 — Lost card", effect: "Compliance stop. No automated retry." },
  { value: "46", label: "46 — Closed account", effect: "Compliance stop. No automated retry." },
  { value: "57", label: "57 — Transaction not permitted", effect: "Compliance stop. No automated retry." },
  { value: "R1", label: "R1 — Customer stopped recurring payments", effect: "Compliance stop. No contact or retry." },
  { value: "R0", label: "R0 — Customer stopped this payment", effect: "Compliance stop. No contact or retry." },
  { value: "R3", label: "R3 — Authorization revoked", effect: "Compliance stop. No contact or retry." },
  { value: "99", label: "Unknown issuer response", effect: "Human review; Revenio does not guess." },
];
const checkoutSignals: Signal[] = [
  { value: "shipping_cost_surprise", label: "Shipping cost changed", effect: "A consented reminder may proceed." },
  { value: "forced_account_creation", label: "Account creation blocked checkout", effect: "A consented return path may be offered." },
  { value: "payment_method_unavailable", label: "Payment method unavailable", effect: "Offer an available payment route." },
  { value: "checkout_form_friction", label: "Checkout form friction", effect: "Record the interruption before messaging." },
  { value: "checkout_page_error", label: "Checkout technical error", effect: "Record the interruption before messaging." },
  { value: "distracted_high_intent", label: "Distracted, high intent", effect: "A consented reminder may proceed." },
  { value: "low_purchase_intent", label: "Low purchase intent", effect: "No automated chasing." },
  { value: "unexpected_processor_response", label: "Unknown processor response", effect: "Human review; the cause is not guessed." },
];
const upiSignals: Signal[] = [
  { value: "U01", label: "U01 — Insufficient funds", effect: "Assess a rail-aware retry." },
  { value: "U02", label: "U02 — Issuer bank unavailable", effect: "Assess a rail-aware retry." },
  { value: "U03", label: "U03 — NPCI technical decline", effect: "Assess a rail-aware retry." },
  { value: "U04", label: "U04 — Beneficiary bank timeout", effect: "Assess a rail-aware retry." },
  { value: "U_REVOKED", label: "Mandate revoked by customer", effect: "Stop immediately; there is no collection authority." },
  { value: "U_PAUSED", label: "Mandate paused by customer", effect: "Stop immediately; there is no collection authority." },
  { value: "U_EXPIRED", label: "Mandate validity expired", effect: "Stop immediately; there is no collection authority." },
  { value: "U99", label: "Unknown UPI return", effect: "Human review; Revenio does not guess." },
];
const nachSignals: Signal[] = [
  { value: "NACH_INSUFFICIENT_FUNDS", label: "Insufficient funds", effect: "A fixed NACH re-presentation may be assessed." },
  { value: "1", label: "Return 1 — Account data correction", effect: "Human review; correct the data before any presentation." },
  { value: "2", label: "Return 2 — Account data correction", effect: "Human review; correct the data before any presentation." },
  { value: "3", label: "Return 3 — Account data correction", effect: "Human review; correct the data before any presentation." },
  { value: "8", label: "Return 8 — Mandate not received", effect: "Stop immediately; no active mandate exists." },
  { value: "9", label: "Return 9 — Miscellaneous", effect: "Human review; the return is not safely classified." },
];

const reviewGuidance: Record<RecoveryDomain, string> = {
  subscription: "Hardship disclosures and unknown issuer responses are owned by a specialist. Approval does not restart automated collection.",
  checkout_abandonment: "Unknown payment signals require a specialist. Missing consent stops customer outreach completely.",
  b2b_receivables: "A specialist approves the final contact tier, and handles disputed or incomplete invoice evidence.",
  mandate_retry: "NACH correction returns and unknown returns require review. UPI AutoPay above ₹15,000 requires a specialist before requesting fresh UPI-PIN authentication.",
};

export function PaymentSimulator({ busy, onSubmit }: Props) {
  const [domain, setDomain] = useState<RecoveryDomain>("subscription");
  const [rail, setRail] = useState<"upi_autopay" | "nach">("upi_autopay");
  const [name, setName] = useState(""); const [customerId, setCustomerId] = useState("");
  const [customerEmail, setCustomerEmail] = useState(""); const [customerPhone, setCustomerPhone] = useState("");
  const [amount, setAmount] = useState("1499"); const [daysOverdue, setDaysOverdue] = useState("30");
  const [failureCode, setFailureCode] = useState("51"); const [response, setResponse] = useState<CustomSimulation["response"]>("recovered");
  const [optIn, setOptIn] = useState(true);
  const options = useMemo(() => responses[domain], [domain]);
  const activeSignals = domain === "subscription" ? subscriptionSignals : domain === "checkout_abandonment" ? checkoutSignals : domain === "mandate_retry" ? rail === "upi_autopay" ? upiSignals : nachSignals : [];
  const selectDomain = (next: RecoveryDomain) => { setDomain(next); setResponse(responses[next][0].value); setFailureCode(next === "subscription" ? "51" : next === "checkout_abandonment" ? "checkout_page_error" : next === "mandate_retry" ? (rail === "upi_autopay" ? "U02" : "NACH_INSUFFICIENT_FUNDS") : ""); };
  const selectRail = (next: "upi_autopay" | "nach") => { setRail(next); setFailureCode(next === "upi_autopay" ? "U02" : "NACH_INSUFFICIENT_FUNDS"); };
  const submit = (event: React.FormEvent) => { event.preventDefault(); onSubmit({ domain_type: domain, customer_name: name, customer_id: customerId || undefined, customer_email: customerEmail || undefined, customer_phone: customerPhone || undefined, amount: Number(amount), failure_code: failureCode || undefined, response, opt_in: optIn, days_overdue: Number(daysOverdue), mandate_rail: rail }); };

  return <section className="payment-simulator"><div className="sim-intro"><p className="eyebrow">NEW RECOVERY CASE</p><h2>Start with a verified payment signal</h2><p>Choose the payment module and its actual failure evidence. Revenio records the permitted path and never books revenue before Razorpay confirms payment.</p><div className="flow-strip"><span>1. Assess the signal</span><span>2. Contact with consent</span><span>3. Confirm payment</span></div><div className="notice"><b>Payment truth comes from Razorpay</b><span>Customer contact can guide the workflow; only a signed payment event marks revenue recovered.</span></div></div><form onSubmit={submit}><fieldset><legend>Choose a recovery module</legend><div className="domain-picker">{domains.map((item) => <button type="button" className={domain === item.value ? "selected" : ""} onClick={() => selectDomain(item.value)} key={item.value}><em>{item.code}</em><b>{item.title}</b><span>{item.description}</span></button>)}</div></fieldset><p className="review-guidance"><b>When a specialist is involved</b>{reviewGuidance[domain]}</p>{domain === "mandate_retry" && <fieldset><legend>Mandate rail</legend><div className="response-picker"><label><input type="radio" checked={rail === "upi_autopay"} onChange={() => selectRail("upi_autopay")} /><span>UPI AutoPay</span></label><label><input type="radio" checked={rail === "nach"} onChange={() => selectRail("nach")} /><span>NACH</span></label></div></fieldset>}<fieldset><legend>Payment and customer details</legend><div className="form-grid"><label>Customer or company<input value={name} placeholder="Customer name or legal business name" onChange={(event) => setName(event.target.value)} required /></label><label>Merchant customer ID<input value={customerId} placeholder="Your internal customer reference" onChange={(event) => setCustomerId(event.target.value)} /></label><label>Email for recovery contact<input type="email" value={customerEmail} placeholder="customer@company.com" onChange={(event) => setCustomerEmail(event.target.value)} /></label><label>Phone for SMS or call<input value={customerPhone} placeholder="+91 98765 43210" onChange={(event) => setCustomerPhone(event.target.value)} /></label><label>Payment amount (INR)<input type="number" min="1" value={amount} onChange={(event) => setAmount(event.target.value)} required /></label>{domain === "b2b_receivables" ? <label>Days overdue<input type="number" min="1" max="365" value={daysOverdue} onChange={(event) => setDaysOverdue(event.target.value)} required /></label> : <label>Recorded payment signal<select value={failureCode} onChange={(event) => setFailureCode(event.target.value)}>{activeSignals.map((signal) => <option value={signal.value} key={signal.value}>{signal.label}</option>)}</select><small className="signal-effect">{activeSignals.find((signal) => signal.value === failureCode)?.effect}</small></label>}</div></fieldset>{domain === "checkout_abandonment" && <label className="consent"><input type="checkbox" checked={optIn} onChange={(event) => setOptIn(event.target.checked)} /> I have recorded customer consent for recovery contact</label>}<fieldset className="policy-branch"><legend>Policy branch to assess</legend><p>This selects a workflow branch for the case. It does not record recovered revenue.</p><div className="response-picker">{options.map((item) => <label key={item.value}><input type="radio" name="response" checked={response === item.value} onChange={() => setResponse(item.value)} /><span>{item.label}</span></label>)}</div></fieldset><button className="primary run-payment" disabled={busy}>{busy ? "Assessing recovery..." : "Create recovery case"}</button></form></section>;
}
