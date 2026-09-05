import { useState } from "react";
import { api, money } from "../api";
import { BulkSimulator } from "../components/BulkSimulator";
import { EventTimeline } from "../components/EventTimeline";
import { PaymentSimulator } from "../components/PaymentSimulator";
import type { CaseDetail, CaseRow, CustomSimulation } from "../types";

type Props = { busy: boolean; selected: CaseDetail | null; cases: CaseRow[]; onSubmit: (request: CustomSimulation) => void; onBulk: (count: number) => void; progress: number; bulkTarget: number; onCase: (caseId: string) => void };

function RazorpayAction({ caseId, amount, onRefresh }: { caseId: string; amount: number; onRefresh: () => void }) {
  const [busy, setBusy] = useState(false);
  const [link, setLink] = useState<string | null>(null);
  const [message, setMessage] = useState("");
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
  return <section className="razorpay-action"><div><p className="eyebrow">COLLECT PAYMENT</p><b>Razorpay test checkout</b><span>{money(amount)} - payment remains pending until Razorpay confirms it.</span></div>{link ? <a className="primary" href={link} target="_blank" rel="noreferrer">Open test checkout</a> : <button className="primary" disabled={busy} onClick={() => void create()}>{busy ? "Creating..." : "Create payment link"}</button>}{message && <small className={link ? "link-note" : "link-error"}>{message}</small>}</section>;
}

export function ExecutionPage({ busy, selected, cases, onSubmit, onBulk, progress, bulkTarget, onCase }: Props) {
  const amount = Number(selected?.case.amount ?? selected?.case.invoice_amount ?? 0);
  return <><section className="run-page-intro"><div><p className="eyebrow">RECOVERY OPERATIONS</p><h2>Create and monitor recovery actions</h2><span>Each case is assessed against its payment signal. Customer contact is recorded, while revenue stays pending until Razorpay confirms payment.</span></div><div className="run-status"><i className="status-dot" /><span>Live case updates enabled</span></div></section><PaymentSimulator busy={busy} onSubmit={onSubmit} /><BulkSimulator busy={busy} progress={progress} target={bulkTarget} onRun={onBulk} /><div className="execution-layout"><section className="panel execution-list"><div className="panel-title"><div><p className="eyebrow">RECENT CASES</p><h2>Recovery activity</h2></div><span>{cases.length}</span></div>{cases.slice(0, 8).map((item) => <button key={item.case_id} onClick={() => onCase(item.case_id)}><strong>{item.customer_name}</strong><small>{item.reason.replaceAll("_", " ")}</small><div><b>{money(item.amount)}</b><span className={`status ${item.status.toLowerCase()}`}>{item.status.replaceAll("_", " ")}</span></div></button>)}{!cases.length && <p className="empty">Create a recovery case to begin.</p>}</section><section className="panel execution-result"><div className="panel-title"><div><p className="eyebrow">CASE ACTIVITY</p><h2>{selected ? String(selected.case.customer_name ?? selected.state.case_id) : "Select a recovery case"}</h2></div>{selected && <span className="status">{String(selected.state.terminal_status ?? "ACTIVE")}</span>}</div>{selected ? <><RazorpayAction caseId={String(selected.state.case_id)} amount={amount} onRefresh={() => onCase(String(selected.state.case_id))} /><EventTimeline events={selected.events} /></> : <p className="empty">Policy decisions, contact receipts, payment confirmation, and review decisions appear here.</p>}</section></div></>;
}
