import { money } from "../api";
import type { AuditEvent } from "../types";

const label = (value: string) => value.replaceAll("_", " ").replace(/([a-z])([A-Z])/g, "$1 $2");
const percent = (value: unknown) => typeof value === "number" ? `${Math.round(value * 100)}%` : "—";
const latestReply = (value: unknown) => String(value ?? "").split(/\n(?:On .+ wrote:|>)/i)[0].trim();

function TechnicalDetails({ response }: { response: unknown }) {
  const value = String(response ?? "").trim();
  if (!value) return null;
  return <details className="technical-details"><summary>Technical delivery details</summary><p>{value}</p></details>;
}

function Evidence({ values }: { values: Record<string, unknown> }) {
  const entries = Object.entries(values).filter(([key, value]) => value !== undefined && value !== null && value !== "" && key !== "source");
  if (!entries.length) return null;
  return <details className="evidence"><summary>Recorded evidence</summary><div>{entries.map(([key, value]) => <span key={key}><small>{label(key)}</small><b>{typeof value === "object" ? JSON.stringify(value) : String(value)}</b></span>)}</div></details>;
}

function DeliveryReceipt({ details }: { details: Record<string, unknown> }) {
  const live = details.mode === "live";
  const submitted = details.effect === "delivery_submitted";
  const channel = label(String(details.channel ?? "contact"));
  const recipient = String(details.recipient ?? "");
  const contact = recipient || "Not provided";
  const status = !live ? "No customer contact was sent" : submitted ? `${channel} submitted` : `${channel} unavailable`;
  const explanation = !live
    ? `This was a portfolio test. Revenio assessed the payment and recorded ${channel.toLowerCase()} as the recommended route, but did not use a customer email address or phone number.`
    : submitted
      ? `${channel} was accepted by the provider for ${contact}. The case stays pending until a customer response or a verified payment event arrives.`
      : "The provider did not accept this contact action. The case remains active; no customer response or payment was assumed.";
  return <div className={`delivery-card ${submitted ? "submitted" : "blocked"}`}><div className="delivery-card-head"><div><small>CONTACT DELIVERY</small><b>{status}</b></div><span className={submitted ? "delivery-live" : live ? "delivery-blocked" : "sandbox"}>{submitted ? "SUBMITTED" : live ? "ACTION NEEDED" : "SIMULATION ONLY"}</span></div><p>{explanation}</p><div className="delivery-meta"><span><small>{live ? "Channel" : "Recommended route"}</small><b>{channel}</b></span><span><small>{live ? "Customer" : "Recipient"}</small><b>{live ? contact : "Not used in this test"}</b></span></div>{live && !submitted ? <TechnicalDetails response={details.response} /> : null}</div>;
}

function EventBody({ event, hasVerifiedPayment }: { event: AuditEvent; hasVerifiedPayment: boolean }) {
  const p = event.payload;
  if (event.event_type === "StopDecision") return <p>{p.should_stop ? `Automation paused: ${label(String(p.stop_reason ?? "safety rule"))}.` : "Compliance checks passed; the permitted recovery path may continue."}</p>;
  if (event.event_type === "Diagnosis") return <div><div className="event-grid"><span><small>Root cause</small><b>{label(String(p.root_cause ?? "under review"))}</b></span><span><small>Confidence</small><b>{percent(p.confidence)}</b></span><span><small>Recovery likelihood</small><b>{percent(p.predicted_recovery_probability)}</b></span></div><Evidence values={(p.raw_signal ?? {}) as Record<string, unknown>} /></div>;
  if (event.event_type === "Decision") return <div className="decision-copy"><b>{label(String(p.action_type ?? "recovery action"))}</b><p>{String(p.reasoning ?? "A policy decision was recorded.")}</p><Evidence values={{ ...((p.action_params ?? {}) as Record<string, unknown>), requires_human_review: p.requires_human_review }} /></div>;
  if (event.event_type === "ExecutionResult") {
    const details = (p.details ?? {}) as Record<string, unknown>;
    return Object.keys(details).length ? <DeliveryReceipt details={details} /> : <p>{p.success ? "The permitted action was recorded." : "The action was stopped by a compliance control."}</p>;
  }
  if (event.event_type === "Outcome") {
    const pending = p.status === "PENDING";
    const simulatedRecovery = Number(p.amount_recovered ?? 0) > 0 && !hasVerifiedPayment;
    return <div><div className="outcome"><b>{simulatedRecovery ? "SIMULATED OUTCOME" : pending ? "PAYMENT PENDING" : label(String(p.status ?? "pending"))}</b><span>{Number(p.amount_recovered ?? 0) > 0 ? hasVerifiedPayment ? `${money(Number(p.amount_recovered))} confirmed by Razorpay.` : `${money(Number(p.amount_recovered))} is a module simulation only. No revenue is booked until Razorpay confirms payment.` : pending ? "No revenue is booked until Razorpay sends a verified payment event." : "No payment has been recovered."}</span></div><Evidence values={(p.details ?? {}) as Record<string, unknown>} /></div>;
  }
  if (event.event_type === "RazorpayPaymentLinkCreated") return <div className="payment-link-event"><b>Secure Razorpay payment link created</b><span>{money(Number(p.amount ?? 0))} · {label(String(p.status ?? "created"))}</span><a href={String(p.short_url)} target="_blank" rel="noreferrer">Open payment link</a></div>;
  if (event.event_type === "RazorpayPaymentEvent") return <div className="payment-link-event"><b>Razorpay payment update</b><span>{label(String(p.event ?? "received"))} · {label(String(p.status ?? "unknown"))} · {money(Number(p.amount ?? 0))}</span></div>;
  if (event.event_type === "RecoveryPaymentFailed") return <p className="attention">Payment was not completed. No revenue was recorded.</p>;
  if (event.event_type === "CustomerEmailReply") return <div className="reply-card"><div className="receipt-head"><b>Customer email reply</b><span className="delivery-live">RECEIVED</span></div><p className="message">{latestReply(p.message) || "The customer replied without readable text."}</p><div className="receipt-grid"><span><small>From</small><b>{String(p.from ?? "Not provided")}</b></span><span><small>Understood as</small><b>{label(String(p.intent ?? "needs human review"))}</b></span><span><small>Next action</small><b>{String(p.action ?? "Review required")}</b></span></div></div>;
  if (event.event_type === "VoiceCallStatus") return <div className="payment-link-event"><b>Hinglish voice call {label(String(p.status ?? "updated"))}</b><span>Twilio call status was recorded for this case.</span></div>;
  if (event.event_type === "CustomerVoiceResponse") return <div className="reply-card"><div className="receipt-head"><b>Customer voice choice</b><span className="delivery-live">RECEIVED</span></div><div className="receipt-grid"><span><small>Key pressed</small><b>{String(p.digits ?? "None")}</b></span><span><small>Understood as</small><b>{label(String(p.intent ?? "unknown"))}</b></span><span><small>Next action</small><b>{String(p.action ?? "Review required")}</b></span></div></div>;
  if (event.event_type === "InboundEmailProcessingFailed") return <p className="attention">The customer reply needs an operator check. Open technical details in the provider log if it persists.</p>;
  if (event.event_type === "PendingHumanReview") return <div className="review-event"><b>Specialist approval required</b><span>{String(p.reason ?? "This recovery action needs approval before it continues.")}</span></div>;
  if (event.event_type === "HumanReviewDecision") return <p>{p.confirmed ? "A specialist approved the next permitted action." : "A specialist stopped automated recovery for this case."}</p>;
  return <p>Recovery event recorded.</p>;
}

export function EventTimeline({ events }: { events: AuditEvent[] }) {
  const hasVerifiedPayment = events.some((event) => event.event_type === "RazorpayPaymentEvent" && (event.payload.event === "payment.captured" || event.payload.event === "payment_link.paid" || event.payload.status === "captured"));
  return <div className="timeline">{events.map((event) => <article key={event.event_id} className={`event ${event.event_type.toLowerCase()}`}><time>{new Date(event.created_at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}</time><div className="event-dot" /><div><div className="event-title"><strong>{label(event.event_type)}</strong><span>{label(event.stage)}</span></div><EventBody event={event} hasVerifiedPayment={hasVerifiedPayment} /></div></article>)}</div>;
}
