import { useMemo, useState } from "react";
import { api } from "../api";
import type { CustomSimulation, RecoveryDomain } from "../types";

type Props = { busy: boolean; onSubmit: (request: CustomSimulation) => void };
type EmailReceipt = { provider: string; recipient: string; response: string; effect: string; provider_message_id?: string };

const domains: { value: RecoveryDomain; code: string; title: string; description: string }[] = [
  { value: "subscription", code: "CARD", title: "Subscription payment", description: "Issuer decline and safe retry" },
  { value: "checkout_abandonment", code: "CHECKOUT", title: "Checkout recovery", description: "Consent-led return to payment" },
  { value: "b2b_receivables", code: "INVOICE", title: "Invoice collection", description: "Email, SMS and approved call" },
  { value: "mandate_retry", code: "UPI", title: "UPI AutoPay", description: "Mandate return assessment" },
];

const responses: Record<RecoveryDomain, { value: CustomSimulation["response"]; label: string }[]> = {
  subscription: [{ value: "recovered", label: "Retry succeeds" }, { value: "lost", label: "Retry is unsuccessful" }, { value: "hardship", label: "Customer asks for hardship support" }],
  checkout_abandonment: [{ value: "recovered", label: "Customer completes payment" }, { value: "lost", label: "Customer does not return" }, { value: "no_response", label: "No response to the message" }],
  b2b_receivables: [{ value: "paid", label: "Customer pays after contact" }, { value: "promise", label: "Customer promises a date" }, { value: "no_response", label: "No response - route to review" }, { value: "needs_human", label: "Customer asks for a specialist" }],
  mandate_retry: [{ value: "recovered", label: "Mandate retry succeeds" }, { value: "lost", label: "Mandate retry fails" }, { value: "no_response", label: "Confirmation is still pending" }],
};

const defaults: Record<RecoveryDomain, string> = { subscription: "51", checkout_abandonment: "checkout_page_error", b2b_receivables: "", mandate_retry: "U02" };
const signals: Record<RecoveryDomain, { value: string; label: string; effect: string }[]> = {
  subscription: [{ value: "51", label: "51 - Insufficient funds", effect: "Policy can assess a bounded retry" }, { value: "05", label: "05 - Do not honour", effect: "Issuer signal is assessed before contact" }, { value: "43", label: "43 - Stolen card", effect: "Compliance stop - no retry" }],
  checkout_abandonment: [{ value: "shipping_cost_surprise", label: "Shipping cost changed", effect: "A consented reminder may proceed" }, { value: "payment_method_unavailable", label: "Payment method unavailable", effect: "Offer a supported payment route" }, { value: "checkout_page_error", label: "Checkout technical error", effect: "Record the interruption before messaging" }, { value: "low_purchase_intent", label: "Low purchase intent", effect: "Do not continue automated chasing" }],
  b2b_receivables: [],
  mandate_retry: [{ value: "U02", label: "U02 - UPI return", effect: "Assess a rail-aware retry" }, { value: "U03", label: "U03 - Mandate issue", effect: "Determine a permitted next action" }, { value: "U04", label: "U04 - Account issue", effect: "Determine a permitted next action" }],
};

function EmailConnectionCheck() {
  const [recipient, setRecipient] = useState("");
  const [busy, setBusy] = useState(false);
  const [receipt, setReceipt] = useState<EmailReceipt | null>(null);
  const send = async () => {
    setBusy(true); setReceipt(null);
    try { setReceipt(await api<EmailReceipt>("/api/integrations/email-test", { method: "POST", body: JSON.stringify({ recipient }) })); }
    catch (error) { setReceipt({ provider: "Resend", recipient, effect: "delivery_blocked", response: error instanceof Error ? error.message : "Email connection check failed." }); }
    finally { setBusy(false); }
  };
  const accepted = receipt?.effect === "delivery_submitted";
  return <section className="email-check"><div><p className="eyebrow">EMAIL CONNECTION</p><b>Check your Resend sender</b><span>Send one connection check and see Resend's exact response.</span></div><div className="email-check-action"><input aria-label="Test email recipient" type="email" value={recipient} placeholder="Your Resend account email" onChange={(event) => setRecipient(event.target.value)} /><button type="button" className="secondary" disabled={!recipient || busy} onClick={() => void send()}>{busy ? "Sending..." : "Send check"}</button></div>{receipt && <div className={accepted ? "email-result accepted" : "email-result blocked"}><b>{accepted ? "Accepted by Resend" : "Not sent"}</b><span>{receipt.response}</span>{receipt.provider_message_id && <small>Provider message ID: {receipt.provider_message_id}</small>}</div>}</section>;
}

export function PaymentSimulator({ busy, onSubmit }: Props) {
  const [domain, setDomain] = useState<RecoveryDomain>("subscription");
  const [name, setName] = useState("");
  const [customerId, setCustomerId] = useState("");
  const [customerEmail, setCustomerEmail] = useState("");
  const [customerPhone, setCustomerPhone] = useState("");
  const [amount, setAmount] = useState("1499");
  const [daysOverdue, setDaysOverdue] = useState("30");
  const [failureCode, setFailureCode] = useState(defaults.subscription);
  const [response, setResponse] = useState<CustomSimulation["response"]>("recovered");
  const [optIn, setOptIn] = useState(true);
  const options = useMemo(() => responses[domain], [domain]);
  const domainSignals = signals[domain];
  const changeDomain = (next: RecoveryDomain) => { setDomain(next); setFailureCode(defaults[next]); setResponse(responses[next][0].value); };
  const submit = (event: React.FormEvent) => {
    event.preventDefault();
    onSubmit({ domain_type: domain, customer_name: name, customer_id: customerId || undefined, customer_email: customerEmail || undefined, customer_phone: customerPhone || undefined, amount: Number(amount), failure_code: failureCode || undefined, response, opt_in: optIn, days_overdue: Number(daysOverdue) });
  };

  return <section className="payment-simulator"><div className="sim-intro"><p className="eyebrow">NEW RECOVERY CASE</p><h2>Start with the payment signal</h2><p>Select the genuine failure reason. The policy records its evidence, allowed action, and customer contact outcome.</p><div className="flow-strip"><span>1. Assess</span><span>2. Contact</span><span>3. Confirm payment</span></div><div className="notice"><b>Revenue stays pending</b><span>Only a signed Razorpay payment event can mark money recovered.</span></div><EmailConnectionCheck /></div><form onSubmit={submit}><fieldset><legend>Choose a recovery module</legend><div className="domain-picker">{domains.map((item) => <button type="button" className={domain === item.value ? "selected" : ""} onClick={() => changeDomain(item.value)} key={item.value}><em>{item.code}</em><b>{item.title}</b><span>{item.description}</span></button>)}</div></fieldset><div className="form-grid"><label>Customer or company<input value={name} placeholder="e.g. Nila Kapoor or Northstar Foods" onChange={(event) => setName(event.target.value)} required /></label><label>Merchant customer ID<input value={customerId} placeholder="e.g. CUST-10482" onChange={(event) => setCustomerId(event.target.value)} /></label><label>Email for recovery contact<input type="email" value={customerEmail} placeholder="customer@example.com" onChange={(event) => setCustomerEmail(event.target.value)} /></label><label>Phone for SMS or call<input value={customerPhone} placeholder="+91 98765 43210" onChange={(event) => setCustomerPhone(event.target.value)} /></label><label>Payment amount (INR)<input type="number" min="1" value={amount} onChange={(event) => setAmount(event.target.value)} required /></label>{domain === "b2b_receivables" ? <label>Days overdue<input type="number" min="1" max="365" value={daysOverdue} onChange={(event) => setDaysOverdue(event.target.value)} required /></label> : <label>Recorded payment signal<select value={failureCode} onChange={(event) => setFailureCode(event.target.value)}>{domainSignals.map((signal) => <option value={signal.value} key={signal.value}>{signal.label}</option>)}</select><small className="signal-effect">{domainSignals.find((signal) => signal.value === failureCode)?.effect}</small></label>}</div>{domain === "checkout_abandonment" && <label className="consent"><input type="checkbox" checked={optIn} onChange={(event) => setOptIn(event.target.checked)} /> Customer consent is available for recovery contact</label>}<fieldset><legend>Customer response to model</legend><div className="response-picker">{options.map((item) => <label key={item.value}><input type="radio" name="response" checked={response === item.value} onChange={() => setResponse(item.value)} /><span>{item.label}</span></label>)}</div></fieldset><button className="primary run-payment" disabled={busy}>{busy ? "Assessing recovery..." : "Assess recovery case"}</button></form></section>;
}
