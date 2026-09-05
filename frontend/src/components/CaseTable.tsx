import { money } from "../api";
import type { CaseRow } from "../types";

export function CaseTable({ cases, onCase }: { cases: CaseRow[]; onCase: (id: string) => void }) {
  return <section className="panel table">
    <div className="panel-title"><h2>Recovery cases</h2><span>{cases.length} cases</span></div>
    <table><thead><tr><th>Customer</th><th>Payment signal</th><th>Module</th><th>Amount</th><th>Case status</th><th /></tr></thead>
      <tbody>{cases.map((item) => <tr key={item.case_id}><td><b>{item.customer_name}</b><small className="case-reference">{item.case_id}</small></td><td>{item.reason.replaceAll("_", " ")}</td><td>{item.domain_type.replaceAll("_", " ")}</td><td>{money(item.amount)}</td><td><span className={`status ${item.status.toLowerCase().replaceAll(":", "-")}`}>{item.status.replace("STOPPED:", "")}</span></td><td><button className="link" onClick={() => onCase(item.case_id)}>Open case</button></td></tr>)}</tbody>
    </table>
    {!cases.length && <p className="empty">No recovery activity yet. Create a recovery case to begin.</p>}
  </section>;
}
