export type Dashboard = {
  money_recovered: number;
  recovery_rate: number;
  active_recoveries: number;
  human_review_count: number;
  total_cases: number;
  domain_breakdown: { domain_type: string; case_count: number }[];
};

export type CaseRow = {
  case_id: string;
  customer_name: string;
  domain_type: string;
  status: string;
  terminal: boolean;
  stage_count: number;
  updated_at: string;
  amount: number;
  reason: string;
};

export type IntegrationStatus = {
  channel_mode: "sandbox" | "live";
  live_delivery_acknowledged: boolean;
  email: { ready: boolean; sender: string | null };
  sms_voice: { ready: boolean };
  razorpay: { ready: boolean; mode: "test" | "live" | null };
};

export type AuditEvent = {
  event_id: number;
  case_id: string;
  stage: string;
  event_type: string;
  payload: Record<string, unknown>;
  created_at: string;
};

export type CaseDetail = {
  state: Record<string, unknown> & { case_id?: string; domain_type?: string; terminal_status?: string; stage_count?: number };
  case: Record<string, unknown> & { customer_name?: string; customer_id?: string; amount?: number; invoice_amount?: number };
  events: AuditEvent[];
};

export type Review = CaseRow & {
  diagnosis: Record<string, unknown>;
  decision: Record<string, unknown>;
};

export type Scenario = "random" | "payment_failure" | "checkout_abandonment" | "overdue_invoice" | "mandate_failure";

export type RecoveryDomain = "subscription" | "checkout_abandonment" | "b2b_receivables" | "mandate_retry";
export type CustomSimulation = {
  domain_type: RecoveryDomain;
  customer_name: string;
  customer_id?: string;
  customer_email?: string;
  customer_phone?: string;
  amount: number;
  failure_code?: string;
  response: "recovered" | "lost" | "paid" | "promise" | "no_response" | "needs_human" | "hardship";
  opt_in: boolean;
  days_overdue: number;
};
