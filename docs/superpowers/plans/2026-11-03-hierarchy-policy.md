# Hierarchy Policy Implementation Plan

Goal: implement the approved phase recovery policy with auditable events and comparable timing.
Architecture: retain controller executors, add durable event recording, update HTTP service and callers.
Tech Stack: Python standard library; existing torch/transformers runtime remains unchanged.
Spec: ../specs/2026-11-03-hierarchy-policy-design.md

## Constraints and review focus
No timeout-triggered replans or fallback. No phase-count cap. No extra GPU dependency.
Review stale callbacks, retries multiplying, reset on repair installation, early success timing,
and failure records incorrectly labelled as robot failure.

- [x] Controller: replace timeout/fallback rules; add symmetric confirmation and phase budget tests.
- [x] Service: remove phase cap/internal retry, retain raw output, expose timing and errors.
- [x] Trace: append requests/results/images durably, export searchable timelines.
- [x] Evaluation: terminate explicitly and measure baseline/hierarchy success consistently.
- [x] Tests: update obsolete policy expectations, run standard-library suite and compilation.
- [x] Documentation: complete Chinese policy inventory and Mermaid mind map; review diff and commit once.
