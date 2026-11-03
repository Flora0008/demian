# Hierarchy policy — approved conversation design

User approved implementation of the consolidated checklist on 2026-11-03.
Approach: modify the existing controller and evaluation path, not replace the architecture.
Rejected: timeout-driven recovery, full-task fallback, arbitrary five-phase cap,
new progress thresholds, extra GPU/model replicas.

Keep monitor cadence 16 steps, minimum phase age 16, confidence .85, two confirmations.
Remove phase timeout. Each phase has two replans; only confirmed advancement resets it,
including advancement within a repair plan. Installing a repair plan does not reset it.
Planner sees full task, remaining-work feedback and complete history. No five-phase cap.
Clear caption only on replan/phase change, keep subtask while waiting, reject stale results.
Caption errors retry asynchronously until success or phase/episode ends.
Planner/monitor each have two total attempts. Exhaustion is service/output error, not physical failure.
Final visual completion without environment success terminates as disagreement; no fallback.
Log all requests, raw outputs, validation failures, images, versions, decisions, timing.
Measure baseline and hierarchy first environment success from the same episode-start boundary.
No new policy thresholds without user approval. Existing interface/generation limits are documented.
