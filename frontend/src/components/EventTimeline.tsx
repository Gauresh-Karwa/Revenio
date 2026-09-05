import { money } from "../api";
import type { AuditEvent } from "../types";

const label = (value: string) => value.replaceAll("_", " ").replace(/([a-z])([A-Z])/g, "$1 $2");
const percent = (value: unknown) => typeof value === "number" ? `${Math.round(value * 100)}%` : "—";

function EventBody({ event }: { event: AuditEvent }) {
  const p = event.payload;
  if (event.event_type === "StopDecision") return <p>{p.should_stop ? `Automation stopped: ${label(String(p.stop_reason ?? "safety rule"))}.` : "Compliance check passed; recovery may continue."}</p>;
  if (event.event_type === "Diagnosis") return <div className="event-grid"><span><small>Root cause</small><b>{label(String(p.root_cause ?? "under review"))}</b></span><span><small>Confidence</small><b>{percent(p.confidence)}</b></span><span><small>Recovery likelihood</small><b>{percent(p.predicted_recovery_probability)}</b></span></div>;
  if (event.event_type === "Decision") return <div><b>{label(String(p.action_type ?? "recovery action"))}</b><p>{String(p.reasoning ?? "A policy decision was recorded.")}</p></div>;
  if (event.event_type === "ExecutionResult") {
    const details = (p.details ?? {}) as Record<string, unknown>;
    if (!Object.keys(details).length) return <p>{p.success ? "Action executed successfully." : "Action was blocked by a compliance control."}</p>;
    const live = details.mode === "live";
    const submitted = details.effect === "delivery_submitted";
    const badge = live ? (submitted ? "LIVE DELIVERY SUBMITTED" : "LIVE DELIVERY BLOCKED") : "SIMULATED — NOT SENT";
    return <div className="receipt"><div className="receipt-head"><b>{String(details.provider ?? "Recovery channel")}</b><span className={live ? (submitted ? "delivery-live" : "delivery-blocked") : "sandbox"}>{badge}</span></div><p className="message">{String(details.message ?? "Recovery action recorded.")}</p><div className="receipt-grid"><span><small>Channel</small><b>{label(String(details.channel ?? "unknown"))}</b></span><span><small>Recipient</small><b>{String(details.recipient ?? "—")}</b></span><span><small>Provider response</small><b>{String(details.response ?? "Awaiting response")}</b></span><span><small>Business effect</small><b>{label(String(details.effect ?? "await_next_step"))}</b></span>{details.reply_to ? <span><small>Reply inbox</small><b>{String(details.reply_to)}</b></span> : null}</div></div>;
  }
  if (event.event_type === "Outcome") { const pending = p.status === "PENDING"; return <div className="outcome"><b>{pending ? "AWAITING RAZORPAY CONFIRMATION" : label(String(p.status ?? "pending"))}</b><span>{Number(p.amount_recovered ?? 0) > 0 ? `${money(Number(p.amount_recovered))} recovered` : pending ? "No revenue is booked. A verified Razorpay payment event is required." : "No payment has been recovered yet."}</span></div>; }
  if (event.event_type === "RazorpayPaymentLinkCreated") return <div className="payment-link-event"><b>Razorpay {String(p.mode ?? "test")} payment link created</b><span>{money(Number(p.amount ?? 0))} · {String(p.status ?? "created")}</span><a href={String(p.short_url)} target="_blank" rel="noreferrer">Open Razorpay checkout</a></div>;
  if (event.event_type === "RazorpayPaymentEvent") return <div className="payment-link-event"><b>Razorpay payment event: {label(String(p.event ?? "received"))}</b><span>Provider status: {String(p.status ?? "unknown")} · {money(Number(p.amount ?? 0))}</span></div>;
  if (event.event_type === "RecoveryPaymentFailed") return <p className="attention">{String(p.message ?? "Razorpay did not confirm payment. No revenue was recovered.")}</p>;
  if (event.event_type === "CustomerEmailReply") return <div className="receipt"><div className="receipt-head"><b>Customer email reply</b><span className="delivery-live">RECEIVED</span></div><p className="message">{String(p.message ?? "")}</p><div className="receipt-grid"><span><small>From</small><b>{String(p.from ?? "Unknown sender")}</b></span><span><small>Understood as</small><b>{label(String(p.intent ?? "needs_human_review"))}</b></span><span><small>Next action</small><b>{String(p.action ?? "Review required")}</b></span></div></div>;
  if (event.event_type === "InboundEmailProcessingFailed") return <p className="attention">Inbound reply needs attention: {String(p.message ?? "Resend could not provide readable email text.")}</p>;
  if (event.event_type === "PendingHumanReview") return <p className="attention">Specialist review required: {String(p.reason ?? "This action needs approval.")}</p>;
  if (event.event_type === "HumanReviewDecision") return <p>{p.confirmed ? "A specialist approved the next recovery action." : "A specialist stopped the automated recovery action."}</p>;
  return <p>Recovery event recorded.</p>;
}

export function EventTimeline({ events }: { events: AuditEvent[] }) {
  return <div className="timeline">{events.map((event) => <article key={event.event_id} className={`event ${event.event_type.toLowerCase()}`}><time>{new Date(event.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time><div className="event-dot" /><div><div className="event-title"><strong>{label(event.event_type)}</strong><span>{label(event.stage)}</span></div><EventBody event={event} /></div></article>)}</div>;
}
