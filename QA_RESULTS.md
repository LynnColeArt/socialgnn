# SocialGNN QA Results (2025-09-25)

## Environment
- Repo commit (pre-change baseline): 959617dd51c9da45bb0d630f493a2634a8392ee4
- Go toolchain: go1.22.2 linux/amd64
- QA agent: Codex CLI (danger-full-access; outbound and loopback networking available)

## Automated Checks
- `go test ./...` → Pass (unit tests plus new HTTP integration coverage under `internal/api`)
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
- Calling `POST /api/train/batch` without any stored training examples now returns HTTP 400 with a clear error payload (`{"error": "no training data available"}`) instead of a generic server error.

## Fixes Implemented
1. **Training batch semantics & error handling** (`internal/api/server.go`, `pkg/engine/engine.go`, `pkg/engine/gnn.go`)
   - Added explicit `batch_size` support, sensible defaults, and epoch looping so training runs even when requested batches exceed the number of stored examples. Introduced the shared `ErrNoTrainingData` sentinel and converts it to an HTTP 400 response with a descriptive JSON payload. Covered by regression tests including `TestEngineTrainBatchHandlesLargeBatch` and integration coverage.
2. **Followback metrics for inactive users** (`pkg/engine/engine.go`)
   - The engine now verifies user existence before returning metrics and no longer treats "all zero" as missing data. Covered by `TestGetFollowbackMetricsReturnsZerosForNewUsers`.
3. **Temporal decay cleanup** (`pkg/engine/graph.go`)
   - Edges older than `maxAge` are deleted, decayed weights below 0.1 are purged, and empty adjacency maps are cleaned up, matching documentation guarantees. Verified by `TestApplyDecayRemovesExpiredAndWeakEdges`.
4. **Public engine package & docs**
   - Moved the GNN engine to `pkg/engine` for external consumption, added HTTP integration tests under `internal/api`, expanded unit coverage in `pkg/engine`, and documented training endpoint defaults/error responses in `MANUAL.md` and `README.md` (library usage section).

## Remaining Observations
- Consider extending integration coverage to spam and comment ranking endpoints to catch future regressions.
