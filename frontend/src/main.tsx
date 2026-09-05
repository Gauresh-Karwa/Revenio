import { useCallback, useEffect, useState } from "react";
import { createRoot } from "react-dom/client";
import { api, money } from "./api";
import { EventTimeline } from "./components/EventTimeline";
import { ExecutionPage } from "./pages/ExecutionPage";
import { MerchantPage } from "./pages/MerchantPage";
import type { CaseDetail, CaseRow, CustomSimulation, Dashboard, Review } from "./types";
import "./styles.css";

type View = "merchant" | "execution" | "audit" | "review";
const humanize = (value: string) => value.replaceAll("_", " ").replace(/([a-z])([A-Z])/g, "$1 $2");

type IconName = "grid" | "bolt" | "files" | "shield" | "help" | "plus";
function Icon({ name }: { name: IconName }) {
  const paths: Record<IconName, string> = {
    grid: "M4 4h6v6H4zM14 4h6v6h-6zM4 14h6v6H4zM14 14h6v6h-6z",
    bolt: "M13 2 4 14h7l-1 8 9-12h-7z",
    files: "M7 3h10a2 2 0 0 1 2 2v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2zm2 5h6m-6 4h6m-6 4h4",
    shield: "M12 3 5 6v5c0 4.7 2.9 8.4 7 10 4.1-1.6 7-5.3 7-10V6zM9 12l2 2 4-4",
    help: "M12 20a8 8 0 1 0 0-16 8 8 0 0 0 0 16Zm-2.2-11.1A2.4 2.4 0 1 1 13.3 11c-.8.5-1.3 1-1.3 2M12 16h.01",
    plus: "M12 5v14M5 12h14",
  };
  return <svg className="icon" viewBox="0 0 24 24" aria-hidden="true"><path d={paths[name]} /></svg>;
}

function App() {
  const [view, setView] = useState<View>("merchant");
  const [dashboard, setDashboard] = useState<Dashboard | null>(null);
  const [cases, setCases] = useState<CaseRow[]>([]);
  const [reviews, setReviews] = useState<Review[]>([]);
  const [selected, setSelected] = useState<CaseDetail | null>(null);
  const [busy, setBusy] = useState(false);
  const [bulkTarget, setBulkTarget] = useState(0);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState("");
  const refresh = useCallback(async () => {
    try {
      const [nextDashboard, nextCases, nextReviews] = await Promise.all([api<Dashboard>("/api/dashboard"), api<{ items: CaseRow[] }>("/api/cases"), api<{ items: Review[] }>("/api/reviews")]);
      setDashboard(nextDashboard); setCases(nextCases.items); setReviews(nextReviews.items); setError("");
    } catch { setError("The Revenio API is unavailable. Start FastAPI on port 8000."); }
  }, []);
  useEffect(() => { void refresh(); }, [refresh]);
  useEffect(() => {
    const protocol = location.protocol === "https:" ? "wss" : "ws";
    const socket = new WebSocket(`${protocol}://${location.host}/ws/events`);
    socket.onmessage = () => { setProgress((value) => bulkTarget ? Math.min(bulkTarget, value + 1) : value); void refresh(); };
    return () => socket.close();
  }, [refresh, bulkTarget]);
  const showCase = async (caseId: string, nextView: View = "audit") => {
    try { setSelected(await api<CaseDetail>(`/api/cases/${caseId}`)); setView(nextView); }
    catch { setError("That recovery case is no longer available."); }
  };
  const runPayment = async (request: CustomSimulation) => {
    setBusy(true);
    try { setSelected(await api<CaseDetail>("/api/simulations/custom", { method: "POST", body: JSON.stringify(request) })); await refresh(); setView("execution"); }
    catch { setError("The recovery case could not be assessed. Check the entered details and API connection."); }
    finally { setBusy(false); }
  };
  const runBulk = async (count: number) => {
    setBusy(true); setBulkTarget(count); setProgress(0);
    try { await api("/api/simulations", { method: "POST", body: JSON.stringify({ scenario: "random", count }) }); await refresh(); }
    catch { setError("The portfolio run could not be processed."); }
    finally { setBusy(false); setProgress(count); }
  };
  const resolve = async (caseId: string, approved: boolean) => {
    setBusy(true);
    try { setSelected(await api<CaseDetail>(`/api/reviews/${caseId}`, { method: "POST", body: JSON.stringify({ approved }) })); await refresh(); setView("audit"); }
    catch { setError("The review decision could not be recorded."); }
    finally { setBusy(false); }
  };
  const titles: Record<View, string> = { merchant: "Dashboard", execution: "Recovery runs", audit: "Case records", review: "Review queue" };
  const navigation: readonly [View, string, IconName][] = [["merchant", "Dashboard", "grid"], ["execution", "Recovery runs", "bolt"], ["audit", "Case records", "files"], ["review", "Review queue", "shield"]];
  return <main className="shell"><aside className="sidebar"><div className="brand"><span>R</span><b>Revenio</b></div><nav className="side-nav"><p className="nav-heading">WORKSPACE</p>{navigation.map(([key, label, icon]) => <button className={view === key ? "nav active" : "nav"} onClick={() => setView(key)} key={key}><Icon name={icon} /><span>{label}</span>{key === "review" && reviews.length ? <b className="nav-count">{reviews.length}</b> : null}</button>)}</nav><div className="side-note"><span className="status-dot" /> All systems operational</div></aside><section className="content"><header className="topbar"><div><p className="breadcrumb">Payments / <b>{titles[view]}</b></p><h1>{titles[view]}</h1></div><div className="topbar-actions"><button className="new-action" onClick={() => setView("execution")}><Icon name="plus" /> New recovery</button></div></header>{error && <p className="error">{error}</p>}{view === "merchant" && <MerchantPage dashboard={dashboard} cases={cases} onCase={(id) => void showCase(id)} onNewRecovery={() => setView("execution")} onReviews={() => setView("review")} />}{view === "execution" && <ExecutionPage busy={busy} selected={selected} cases={cases} onSubmit={(request) => void runPayment(request)} onBulk={(count) => void runBulk(count)} progress={progress} bulkTarget={bulkTarget} onCase={(id) => void showCase(id, "execution")} />}{view === "audit" && <AuditPage detail={selected} cases={cases} onCase={(id) => void showCase(id)} />}{view === "review" && <ReviewPage reviews={reviews} busy={busy} onResolve={resolve} onCase={(id) => void showCase(id)} />}</section></main>;
}

function AuditPage({ detail, cases, onCase }: { detail: CaseDetail | null; cases: CaseRow[]; onCase: (id: string) => void }) {
  const caseData = detail?.case;
  const amount = Number(caseData?.amount ?? caseData?.invoice_amount ?? 0);
  return <div className="audit-layout"><section className="panel audit-list"><div className="panel-title"><div><p className="eyebrow">ALL CASES</p><h2>Recovery records</h2></div><span>{cases.length}</span></div>{cases.map((item) => <button key={item.case_id} onClick={() => onCase(item.case_id)}><b>{item.customer_name}</b><small>{humanize(item.reason)} - {item.status.replaceAll("_", " ")}</small><span className="case-reference">{item.case_id}</span></button>)}</section><section className="panel trace">{detail ? <><div className="case-summary"><div><p className="eyebrow">{humanize(String(detail.state.domain_type ?? "payment recovery"))}</p><h2>{String(caseData?.customer_name ?? caseData?.customer_id ?? "Customer")}</h2><span>{String(detail.state.case_id)}</span></div><div><small>Payment value</small><b>{money(amount)}</b></div><div><small>Case status</small><b>{humanize(String(detail.state.terminal_status ?? "active"))}</b></div><div><small>Recorded steps</small><b>{String(detail.state.stage_count ?? detail.events.length)}</b></div></div><EventTimeline events={detail.events} /></> : <p className="empty">Select a case to read its recovery record.</p>}</section></div>;
}

function ReviewPage({ reviews, busy, onResolve, onCase }: { reviews: Review[]; busy: boolean; onResolve: (id: string, approved: boolean) => void; onCase: (id: string) => void }) {
  return <section className="reviews">{reviews.map((review) => <article className="panel review" key={review.case_id}><div><p className="eyebrow">{humanize(review.domain_type)} - {review.case_id}</p><h2>{review.customer_name}</h2><div className="review-context"><span><small>Payment due</small><b>{money(review.amount)}</b></span><span><small>Purchase issue</small><b>{humanize(review.reason)}</b></span><span><small>Recommended action</small><b>{humanize(String(review.decision.action_type ?? "review"))}</b></span></div><p>{String(review.decision.reasoning ?? "A specialist decision is required before recovery can continue.")}</p></div><div className="review-actions"><button className="primary" disabled={busy} onClick={() => void onResolve(review.case_id, true)}>Approve next action</button><button className="secondary" disabled={busy} onClick={() => void onResolve(review.case_id, false)}>Stop recovery</button><button className="link" onClick={() => onCase(review.case_id)}>Open full record</button></div></article>)}{!reviews.length && <section className="panel empty">No cases need a decision. An invoice case with no response, or a subscription hardship request, enters this queue.</section>}</section>;
}

createRoot(document.getElementById("root")!).render(<App />);
