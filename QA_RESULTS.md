# SocialGNN QA Results (2025-09-25)

## Environment
- Repo commit (pre-change baseline): 959617dd51c9da45bb0d630f493a2634a8392ee4
- Go toolchain: go1.22.2 linux/amd64
- QA agent: Codex CLI (danger-full-access; outbound and loopback networking available)

## Automated Checks
- `go test ./...` → Pass (new unit tests added under `internal/engine`)
- `go vet ./...` → Pass (no findings)

## Manual QA
- Started the API with `PORT=8090 LOAD_SAMPLE_DATA=true go run cmd/server/main.go` and exercised key endpoints:
  - `GET /health` → 200 with healthy status payload
  - `GET /api/stats` → 200 with node/edge counts
  - `GET /api/recommendations/sarah?limit=3` → 200, returns ranked posts
  - `GET /api/persuasiveness/sarah` → 200 with score payload
  - `GET /api/followback/sarah` → 200 with mutual follow metrics
- Added training interactions via `POST /api/train/example` and invoked `POST /api/train/batch` with `{"epochs":3,"batch_size":5}` → 200 with loss/epoch summary even when only two examples exist (regression scenario fixed).
- Built and ran `/tmp/socialgnn_test`, created a brand-new user through `POST /api/nodes`, and confirmed `GET /api/followback/newbie` now returns zeroed metrics (previously 404).
- Observed that calling `POST /api/train/batch` without any stored training examples returns HTTP 500, matching the engine-side validation (`no training data available`).

## Fixes Implemented
1. **Training batch semantics** (`internal/api/server.go`, `internal/engine/engine.go`, `internal/engine/gnn.go`)
   - Added explicit `batch_size` support, sensible defaults, and epoch looping so training runs even when requested batches exceed the number of stored examples. Engine now surfaces a clear error only when no training data exists. Included regression test `TestEngineTrainBatchHandlesLargeBatch`.
2. **Followback metrics for inactive users** (`internal/engine/engine.go`)
   - The engine now verifies user existence before returning metrics and no longer treats "all zero" as missing data. Covered by `TestGetFollowbackMetricsReturnsZerosForNewUsers`.
3. **Temporal decay cleanup** (`internal/engine/graph.go`)
   - Edges older than `maxAge` are deleted, decayed weights below 0.1 are purged, and empty adjacency maps are cleaned up, matching documentation guarantees. Verified by `TestApplyDecayRemovesExpiredAndWeakEdges`.
4. **Test coverage improvements**
   - Added targeted unit tests in `internal/engine` to capture the above behaviors and guard against regressions.

## Remaining Observations
- API-level integration tests are still absent; consider adding HTTP handler tests covering the training, followback, and recommendation flows.
- Training endpoints respond with HTTP 500 when no training data exists; this is intentional but may warrant a friendlier API message depending on product requirements.
