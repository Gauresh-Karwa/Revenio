type Props = { busy: boolean; progress: number; target: number; onRun: (count: number) => void };

export function BulkSimulator({ busy, progress, target, onRun }: Props) {
  const displayed = Math.min(progress, target);
  return <section className="panel bulk-simulator"><div><p className="eyebrow">PORTFOLIO TEST</p><h2>Run a varied payment portfolio</h2><p>Fresh payment failures, checkout drop-offs, invoices, and mandate returns. No recipients are contacted in this mode.</p></div><div className="bulk-actions">{[10, 100].map((count) => <button className={count === 100 ? "primary" : "secondary"} disabled={busy} onClick={() => onRun(count)} key={count}>Run {count} cases</button>)}</div>{busy && <div className="progress-wrap"><div><span>Cases processed</span><b>{displayed} / {target}</b></div><progress value={displayed} max={target} /></div>}</section>;
}
