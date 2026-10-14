# tdeez-ai-backend

Backend for TDeez's AI design and visualization of custom live-edge slab furniture. It serves the
existing React frontend; **all AI work goes through Amazon Bedrock** and the frontend never calls
AWS AI services directly.

* Python 3.12 · FastAPI · Pydantic v2 · boto3 · Mangum (Lambda) or uvicorn (container)
* Every external dependency (Bedrock, S3, DynamoDB, OpenSearch, SES) is behind an interface with an
  in-memory/mock implementation, so the project **runs and tests pass with no AWS account**.

> **Contract status:** `apiEndpoints.ts` and `src/types/*.ts` were not available when this was built.
> The endpoints and shapes below are a **proposed contract** (see [Proposed contract](#proposed-contract-to-reconcile-with-the-frontend)).
> Reconcile them with the frontend, then rename routes/fields in `src/tdeez_ai/api` and `models`.

## Quick start (mock mode, no AWS)

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"      # or: python3.12 -m venv .venv && pip install -e ".[dev]"
cp .env.example .env                                      # USE_MOCK=true by default
uvicorn tdeez_ai.main:app --reload                        # http://localhost:8000/docs
```

In mock mode:

* Authenticate with `Authorization: Bearer mock-<name>` (any `mock-` token is a distinct user). This
  verifier is only wired when `USE_MOCK=true`.
* Inventory is auto-seeded at startup (`SEED_ON_STARTUP`).
* "Presigned" upload URLs point at a dev-only `PUT /mock-upload/{key}` (accepts only keys that were issued),
  and image URLs at `GET /mock-files/{key}`. Both routes 404 outside mock mode.
* Bedrock is replaced by deterministic fakes (tiny PNGs, fixed species/confidence).

```bash
ruff check . && ruff format --check . && mypy && pytest   # all offline
python scripts/seed_inventory.py                          # seed + embed sample inventory
```

## Switching to real Bedrock / AWS

1. Deploy `infra/template.yaml` (see [Infrastructure](#infrastructure)) or create the equivalent resources.
2. Enable model access in the Bedrock console for **Nova Canvas, Nova Pro (or Lite) and Titan Multimodal Embeddings G1** in your region.
3. Set `USE_MOCK=false` and every variable in *Real-AWS resources* below. Startup fails with a clear
   message listing any missing variable names.
4. Credentials come from the default AWS provider chain / the Lambda or task role. Never put keys in `.env`.
5. Seed inventory with real photos so similarity is meaningful:
   `python scripts/seed_inventory.py --images-dir ./sample_images` (files named `slab-001.jpg` ... `slab-008.jpg`).

Run as Lambda: handler `tdeez_ai.handler.handler` (Mangum). As a container: `docker build . && docker run -p 8000:8000 --env-file .env <image>`.

## Environment variables

| Variable | Default | Purpose |
|---|---|---|
| `USE_MOCK` | `true` | In-memory repos + fake Bedrock + mock auth |
| `AWS_REGION` | `us-east-1` | Region for Bedrock/S3/DynamoDB/SES/Cognito |
| `CANVAS_MODEL_ID` | `amazon.nova-canvas-v1:0` | Image generation/editing, background removal |
| `VISION_MODEL_ID` | `amazon.nova-pro-v1:0` | Multimodal understanding via Converse (set a Nova Lite ID to save cost) |
| `EMBEDDING_MODEL_ID` / `EMBEDDING_DIMENSIONS` | `amazon.titan-embed-image-v1` / `1024` | Similarity embeddings (256, 384 or 1024) |
| `BEDROCK_READ_TIMEOUT_S` / `BEDROCK_CONNECT_TIMEOUT_S` | `120` / `5` | Per-call timeouts |
| `BEDROCK_MAX_RETRIES` / `BEDROCK_BACKOFF_BASE_S` | `3` / `0.5` | Bounded retry with exponential backoff + jitter |
| `MAX_IMAGES_PER_REQUEST` / `MAX_IMAGE_BYTES` | `3` / `3750000` | Cost guardrails (3.75 MB is the Converse per-image limit) |
| `SPECIES_REVIEW_THRESHOLD` | `0.6` | Below this, `needsReview=true` |
| `ROOM_MIN_CONFIDENCE` | `0.5` | Below this, room photo → `fallbackToStandalone=true` |
| `PRICING_RULES_PROVIDED` / `PRICING_RULES_PATH` | `false` / – | Price range returned only when true and a rules file is given |
| `UPLOAD_URL_EXPIRY_S` / `READ_URL_EXPIRY_S` | `300` / `300` | Presigned URL lifetimes (max 900) |
| `CORS_ALLOW_ORIGINS` | `[]` | JSON list of allowed origins |
| *Real-AWS resources* | – | `S3_BUCKET`, `CREATIONS_TABLE`, `SLABS_TABLE`, `PROFILES_TABLE`, `QUOTES_TABLE`, `OPENSEARCH_ENDPOINT`, `OPENSEARCH_INDEX`, `SES_SENDER`, `QUOTE_INBOX`, `COGNITO_USER_POOL_ID`, `COGNITO_APP_CLIENT_ID` |

Pricing rules file (only if you have real rules): `{"currency":"USD","base":{"dining_table":3000,"coffee_table":1200,"desk":1800},"epoxy_surcharge":400,"spread_pct":0.15}`.

## Endpoints → feature IDs

All routes require `Authorization: Bearer <Cognito access or ID token>` except `/health` and `/share/{token}`.
Responses use `{ "isSuccess": bool, "message": str, "data": T }` (camelCase fields).

| Method & path | Feature IDs | Notes |
|---|---|---|
| *(Cognito, no backend route)* | POC-001 – POC-005 | Sign up/in/reset/out handled by Cognito; backend only verifies the JWT |
| `GET`/`PUT /profile` | POC-031 | name, email, phone, `extras` (DynamoDB) |
| `POST /creations`, `GET /creations`, `GET`/`PUT`/`DELETE /creations/{id}` | POC-006, TBC-013 | Customer-scoped CRUD; images in S3 via presigned GET |
| `POST`/`DELETE /creations/{id}/share` | Public share | Mint / revoke an unguessable token |
| `GET /share/{token}` | Public share | **No auth**; read-only projection without owner or storage keys |
| `POST /slabs/upload-url` | TBC-008 | Presigned PUT; JPEG/PNG only, size limit |
| `GET /slabs?species=&availability=` | TBC-008 | Inventory browse |
| `POST /slabs/select`, `GET /session` | TBC-008, TBC-011 | Active slab + furniture type in session metadata |
| `POST /slabs/similar` | TBC-008 | Titan embedding → k-NN; `topK` 3–5 with scores |
| `POST /slabs/analyze` | TBC-010 | BG removal, outline, relative proportions, defects, species + confidence, summary |
| `POST /concepts/generate` | TBC-011 | 2–3 concepts; `regenerate` bumps the seed; changing `furnitureType` keeps the slab |
| `POST /rooms/upload-url`, `/rooms/validate`, `/rooms/visualize` | TBC-009 | Suitability check, fallback flag, in-room draft |
| `POST /customization/validate`, `/recommend-legs`, `/render` | TBC-012 | Deterministic ranges; AI leg suggestion; preview re-render |
| `POST /quotes` | TBC-013 | Persist + SES email to the TDeez inbox; returns `TD-YYYYMMDD-XXXXXX` |
| `GET /health` | – | Unauthenticated liveness |

### Proposed contract (to reconcile with the frontend)

Everything above is proposed. Open questions to confirm against `apiEndpoints.ts` / `src/types`:

* Route names and a possible `/api/v1` prefix; HTTP verbs for update (`PUT` vs `PATCH`).
* Whether the frontend uploads via presigned PUT (assumed) or multipart to the API.
* Field names/casing (assumed camelCase) and enum spellings (`dining_table`, `black_walnut`, ...).
* Species list (assumed: black walnut, big-leaf maple, white oak, cherry, acacia) and customization enums/ranges (`services/customization.py`).
* Dimension units (assumed inches).

## Hard rules and where they are enforced

| Rule | Enforcement | Tests |
|---|---|---|
| 1. AI output is a draft | `disclaimer` on every AI response; `services/text_guard.py` strips measurements/structural claims/prices from model text; proportions are ratios unless inventory dimensions exist; `priceRange` only with pricing rules | `test_species.py`, `test_validation.py` |
| 2. Confidence / fallback | `SlabAnalysisService` (`needsReview`), `RoomService` (`fallbackToStandalone`, never a guessed image) | `test_species.py`, `test_room.py` |
| 3. Preserve slab geometry | `bedrock/prompts.py` `GEOMETRY_CLAUSE` + negative text; high `similarityStrength` | `test_concepts.py` |
| 4. Privacy | `core/logging.py` field allowlist; error handlers return field names only; hashed storage prefixes; private SSE bucket + short-lived presigned URLs | `test_privacy.py`, `test_envelope.py` |
| 5. Customer scoping | Identity only from the verified JWT; every repo call takes the owner; storage keys must be under the caller's prefix | `test_auth.py` |
| 6. Bedrock hygiene | `bedrock/client.py` timeouts, bounded retry/backoff; `bedrock/errors.py` mapping; `bedrock/guard.py` | `test_bedrock.py` |
| 7. Typing / secrets | `mypy --strict`; no Any in `src`; `.env.example` only | CI commands above |

## Architecture

```
src/tdeez_ai/
  api/          routers (one per domain) + JWT dependency
  services/     business logic, framework-agnostic
  bedrock/      client (retry/timeouts), Nova Canvas, Nova vision (Converse), Titan, prompts, guard, mocks
  repositories/ interfaces (base.py) + in-memory, DynamoDB, S3, OpenSearch, SES
  models/       Pydantic schemas
  core/         config, logging, errors, auth, keys, images
  container.py  wiring (mock vs AWS); app.py factory; handler.py (Lambda)
```

DynamoDB key design: `creations` pk `USER#<sub>` / sk `CREATION#<id>` (+ sparse GSI `shareToken-index`);
`slabs` pk `SLAB#<id>` / sk `META` (+ GSI `species-availability-index`); `profiles` pk `USER#<sub>` /
sk `PROFILE` or `SESSION`; `quotes` pk `USER#<sub>` / sk `QUOTE#<ref>`.

Model usage was checked against the Bedrock docs: Nova Canvas task types and payloads
(`TEXT_IMAGE`, `IMAGE_VARIATION`, `INPAINTING`, `OUTPAINTING`, `BACKGROUND_REMOVAL`; no width/height
for inpaint/outpaint; moderated results come back through an `error` field) and Titan Multimodal
Embeddings G1 (`inputImage` + `embeddingConfig.outputEmbeddingLength` ∈ {256, 384, 1024}).
Not verified against the docs: the Canvas seed upper bound and the exact input-image pixel limits;
OpenSearch Serverless faiss `l2` space support. Confirm these in your account before relying on them.

## Infrastructure

`infra/template.yaml` is an AWS SAM outline: HTTP API + Cognito JWT authorizer (public `/share/{token}`
and `/health`), Lambda, four DynamoDB tables, private SSE S3 bucket (TLS-only policy, lifecycle), OpenSearch
Serverless vector collection with encryption/network/data-access policies, SES identity, and
least-privilege IAM (`bedrock:InvokeModel` limited to the three model ARNs). **It has not been deployed**;
review and `sam validate` before use. Note that the table GSIs and the vector index mapping
(`repositories/opensearch.py`, created on startup) must match the code.

## Added dependencies (beyond the agreed stack)

| Package | Why |
|---|---|
| `PyJWT[crypto]` | Verify Cognito RS256 JWTs against the user pool JWKS |
| `opensearch-py` | OpenSearch Serverless k-NN client (SigV4 signing built in) |
| `httpx` (dev) | Required by FastAPI's `TestClient` |
| `pillow`, `numpy` (optional `evals` extra) | Mask IoU in the background-removal eval only |

`src/requirements.txt` duplicates the runtime list for `sam build`; keep it in sync with `pyproject.toml`.

## Known limitations

* **Accuracy targets are unmeasured.** ≥ 80 % Top-5 similarity and ≥ 90 % background removal need a
  curated labelled set and real Bedrock. `evals/` has skeletons only; the unit tests prove plumbing, not model quality.
* **Room visualization is prompt-driven.** Nova Canvas inpainting places a *described* table in the room
  photo; it does not composite the actual slab or concept image. Treat it as an illustrative draft.
* **Concept fidelity is probabilistic.** Geometry is preserved by instruction, high similarity strength and
  negative prompts, but the model can still drift; nothing verifies the outline automatically.
* **Latency vs. API Gateway.** HTTP APIs time out at 30 s; generating 2–3 Canvas images (run in parallel) or
  analyzing a slab can exceed that. For production use the container path (ALB/App Runner) or add an async job
  pattern (submit + poll) — not implemented.
* **Presigned PUT cannot enforce size.** Declared size is validated up front; actual bytes are re-checked
  (magic bytes + size) when read, and invalid owned uploads are deleted. Add an S3 event/lifecycle rule for orphans.
* **Quote email failure** does not lose the request: it is stored with `emailSent=false` and the customer still
  gets a reference. There is no retry worker yet.
* **Deleting a creation** removes metadata only; images expire via the bucket lifecycle rule.
* **Species list, enums and dimension ranges are assumptions** pending product sign-off.
* OpenSearch Serverless vector collections do not support custom document ids, so re-seeding adds duplicates;
  recreate the index before re-seeding.
* DynamoDB, S3, OpenSearch, SES and Cognito (JWKS) implementations are not exercised by the offline test suite
  (only Bedrock is covered with botocore Stubber); verify them in a dev account.
