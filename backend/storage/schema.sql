
-- Architecture doc 9.1: the audit log is the source of truth. events is
-- append-only (9.6) -- no UPDATE, no DELETE, ever, at the application layer.
-- Architecture doc 9.6: index on case_id for the audit log, and on
-- (status, domain_type, updated_at) for whatever queries the merchant feed
-- and human-review queue run constantly. This schema realizes both:
--
--   events      -- the append-only source of truth (mirrors backend/core/events.py's Event)
--   case_state  -- a MATERIALIZED, denormalized read model, updated in the
--                  SAME transaction as every events insert (9.1's "single
--                  atomic database transaction"). This is what makes
--                  derive_state() an indexed point lookup instead of a full
--                  replay of every event on every read -- the replay-based
--                  derive_state() in the in-memory EventStore was fine at
--                  in-memory scale, but doesn't hold up as "every list-facing
--                  query is paginated, never a full scan" (9.6) once there
--                  are millions of events. case_state is NOT a second source
--                  of truth in the dual-write sense that 9.1 warns against --
--                  it's derived FROM events, written in the same transaction,
--                  and could be rebuilt from events at any time by replay if
--                  it were ever lost or found to have drifted.

CREATE TABLE IF NOT EXISTS events (
    event_id     BIGSERIAL PRIMARY KEY,
    case_id      TEXT NOT NULL,
    domain_type  TEXT NOT NULL,
    stage        TEXT NOT NULL,
    event_type   TEXT NOT NULL,
    payload      JSONB NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    customer_id  TEXT
);

-- Every get_events(case_id) call in the codebase filters on case_id alone
-- and needs insertion order back -- event_id (a BIGSERIAL) already IS
-- insertion order, so no separate timestamp sort is needed at read time.
CREATE INDEX IF NOT EXISTS idx_events_case_id
    ON events (case_id, event_id);

-- get_customer_case_history(customer_id, exclude_case_id) is the
-- cross-case query subscription's diagnose() depends on for
-- customer_recent_failure_pressure -- this is its index.
CREATE INDEX IF NOT EXISTS idx_events_customer_id
    ON events (customer_id, event_id)
    WHERE customer_id IS NOT NULL;

CREATE TABLE IF NOT EXISTS case_state (
    case_id          TEXT PRIMARY KEY,
    domain_type      TEXT NOT NULL,
    -- 'ACTIVE' | 'STOPPED:<StopReason>' | 'RECOVERED' | 'LOST'
    -- Mirrors the exact terminal_status strings the in-memory EventStore
    -- computes in derive_state(), so nothing downstream (Orchestrator,
    -- tests, a future frontend) needs to know which store implementation
    -- is behind the interface.
    status           TEXT NOT NULL DEFAULT 'ACTIVE',
    terminal         BOOLEAN NOT NULL DEFAULT FALSE,
    stage_count      INT NOT NULL DEFAULT 0,
    last_stage       TEXT,
    last_event_type  TEXT,
    updated_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Architecture doc 9.6, verbatim: "Indexes on (status, domain_type,
-- updated_at) for the case table (what the merchant feed and
-- human-review queue query constantly)."
CREATE INDEX IF NOT EXISTS idx_case_state_status_domain_updated
    ON case_state (status, domain_type, updated_at);

-- Architecture doc 9.5: "policy updates aren't applied inline per-case;
-- outcomes are queued... and a single dedicated consumer applies them to
-- the policy sequentially -- one writer, no races." This table is that
-- policy's DURABLE state, so a Celery worker can be killed and restarted
-- without losing bandit history (the in-memory-only LearningCore from
-- Step 6 loses everything on process restart -- fine for tests, not fine
-- once this runs as a long-lived worker process). One row per domain_type,
-- state JSONB is exactly BanditPolicy.to_dict()'s output -- no translation
-- layer, no second serialization format to keep in sync with the first.
CREATE TABLE IF NOT EXISTS bandit_policy_state (
    domain_type  TEXT PRIMARY KEY,
    state        JSONB NOT NULL,
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

