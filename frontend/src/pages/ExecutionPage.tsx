import { useState } from "react";
import { api, money } from "../api";
import { BulkSimulator } from "../components/BulkSimulator";
import { EventTimeline } from "../components/EventTimeline";
import { PaymentSimulator } from "../components/PaymentSimulator";
import type { AuditEvent, CaseDetail, CaseRow, CustomSimulation, IntegrationStatus } from "../types";

type Props = { busy: boolean; selected: CaseDetail | null; cases: CaseRow[]; integrations: IntegrationStatus | null; onSubmit: (request: CustomSimulation) => void; onBulk: (count: number) => void; progress: number; bulkTarget: number; onCase: (caseId: string) => void };

function hasVerifiedPayment(events: AuditEvent[]) {
  return events.some((event) => event.event_type === "RazorpayPaymentEvent" && (event.payload.event === "payment.captured" || event.payload.event === "payment_link.paid" || event.payload.status === "captured"));
}

function RazorpayAction({ caseId, amount, events, onRefresh }: { caseId: string; amount: number; events: AuditEvent[]; onRefresh: () => void }) {
  const [busy, setBusy] = useState(false);
  const [link, setLink] = useState<string | null>(null);
  const [message, setMessage] = useState("");
  const existing = [...events].reverse().find((event) => event.event_type === "RazorpayPaymentLinkCreated");
  const existingLink = typeof existing?.payload.short_url === "string" ? existing.payload.short_url : null;
  const paymentActivity = events.filter((event) => ["RazorpayPaymentLinkCreated", "RazorpayPaymentEvent", "RecoveryPaymentFailed"].includes(event.event_type));
  const paymentFailed = paymentActivity.at(-1)?.event_type === "RecoveryPaymentFailed";
  const recovered = hasVerifiedPayment(events);
  const create = async () => {
    setBusy(true); setMessage("");
    try {
      const result = await api<{ short_url?: string; status?: string }>(`/api/cases/${caseId}/payment-link`, { method: "POST" });
      setLink(result.short_url ?? null);
      setMessage(result.short_url ? "Link created. Complete or fail the test checkout; Razorpay's verified webhook will update this case." : "Razorpay did not return a checkout URL.");
      onRefresh();
    } catch (error) { setMessage(error instanceof Error ? error.message : "Unable to create the Razorpay payment link."); }
    finally { setBusy(false); }
  };
  if (recovered) return <section className="razorpay-action confirmed"><div><p className="eyebrow">PAYMENT COLLECTION</p><b>Payment confirmed</b><span>{money(amount)} has been confirmed by Razorpay and is included in recovered revenue.</span></div><span className="status recovered">Recovered</span></section>;
  const visibleLink = link ?? existingLink;
  return <section className="razorpay-action"><div><p className="eyebrow">PAYMENT COLLECTION</p><b>{paymentFailed ? "Payment needs a new attempt" : visibleLink ? "Secure payment link is ready" : "Create a secure payment link"}</b><span>{money(amount)} remains pending until Razorpay confirms payment. {paymentFailed ? "The previous payment was not completed." : ""}</span></div>{visibleLink ? <a className="primary" href={visibleLink} target="_blank" rel="noreferrer">Open payment link</a> : <button className="primary" disabled={busy} onClick={() => void create()}>{busy ? "Creating..." : "Create payment link"}</button>}{message && <small className={link ? "link-note" : "link-error"}>{message}</small>}</section>;
}

function CaseSnapshot({ detail, amount }: { detail: CaseDetail; amount: number }) {
  const events = detail.events;
  const latest = [...events].reverse().find((event) => ["PendingHumanReview", "CustomerEmailReply", "CustomerVoiceResponse", "RazorpayPaymentEvent", "Outcome"].includes(event.event_type));
  const caseStatus = String(detail.state.terminal_status ?? "ACTIVE");
  const paymentConfirmed = hasVerifiedPayment(events);
  const signal = events.find((event) => event.event_type === "Diagnosis")?.payload.root_cause;
  const nextAction = latest?.event_type === "PendingHumanReview" ? "Specialist approval required" : latest?.event_type === "CustomerEmailReply" ? "Review the customer reply" : paymentConfirmed ? "Payment confirmed" : caseStatus.toUpperCase() === "LOST" ? "Recovery was not completed" : "Await Razorpay payment confirmation";
  return <section className="case-snapshot"><div className="snapshot-main"><p className="eyebrow">CASE OVERVIEW</p><b>{String(detail.case.customer_name ?? detail.case.customer_id ?? "Customer")}</b><span>{String(detail.state.case_id)}</span></div><div><small>Payment value</small><b>{money(amount)}</b></div><div><small>Payment signal</small><b>{String(signal ?? "Under assessment").replaceAll("_", " ")}</b></div><div><small>Next action</small><b>{nextAction}</b></div><div className="snapshot-contact"><small>Contact record</small><b>{String(detail.case.customer_email ?? "Not provided")} · {String(detail.case.customer_phone ?? "Not provided")}</b></div></section>;
}

export function ExecutionPage({ busy, selected, cases, integrations, onSubmit, onBulk, progress, bulkTarget, onCase }: Props) {
  const amount = Number(selected?.case.amount ?? selected?.case.invoice_amount ?? 0);
  const serviceMessage = !integrations ? "Checking live service configuration" : integrations.channel_mode === "live" && integrations.live_delivery_acknowledged ? "Live delivery is enabled for configured channels" : "Portfolio simulation mode — no customer is contacted by bulk tests";
  const selectedStatus = selected ? hasVerifiedPayment(selected.events) ? "RECOVERED" : String(selected.state.terminal_status ?? "ACTIVE") === "RECOVERED" ? "SIMULATION ONLY" : String(selected.state.terminal_status ?? "ACTIVE") : null;
  return <><section className="run-page-intro"><div><p className="eyebrow">RECOVERY OPERATIONS</p><h2>Create, collect, and resolve recovery cases</h2><span>Each case follows its payment evidence. Customer contact can guide the workflow; Razorpay remains the only source of recovered-revenue truth.</span></div><div className="run-status"><i className={`status-dot ${integrations?.channel_mode === "live" && integrations.live_delivery_acknowledged ? "" : "caution"}`} /><span>{serviceMessage}</span></div></section><PaymentSimulator busy={busy} onSubmit={onSubmit} /><BulkSimulator busy={busy} progress={progress} target={bulkTarget} onRun={onBulk} /><div className="execution-layout"><section className="panel execution-list"><div className="panel-title"><div><p className="eyebrow">RECENT CASES</p><h2>Recovery activity</h2></div><span>{cases.length}</span></div>{cases.slice(0, 8).map((item) => <button key={item.case_id} onClick={() => onCase(item.case_id)}><strong>{item.customer_name}</strong><small>{item.reason.replaceAll("_", " ")}</small><div><b>{money(item.amount)}</b><span className={`status ${item.status.toLowerCase()}`}>{item.status.replaceAll("_", " ")}</span></div></button>)}{!cases.length && <p className="empty">Create a recovery case to begin.</p>}</section><section className="panel execution-result"><div className="panel-title"><div><p className="eyebrow">CASE ACTIVITY</p><h2>{selected ? String(selected.case.customer_name ?? selected.state.case_id) : "Select a recovery case"}</h2></div>{selected && <span className="status">{selectedStatus}</span>}</div>{selected ? <><CaseSnapshot detail={selected} amount={amount} /><RazorpayAction caseId={String(selected.state.case_id)} amount={amount} events={selected.events} onRefresh={() => onCase(String(selected.state.case_id))} /><EventTimeline events={selected.events} /></> : <p className="empty">Select a case to see payment status, customer response, and the next permitted action.</p>}</section></div></>;
}
