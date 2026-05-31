# Handoff: sub2api Gemini **pro-tier** concurrency / account-routing investigation

**Date:** 2026-05-31
**Author:** main session (Gemini FP-review calibration)
**For:** a parallel investigation agent
**Status:** OPEN — needs investigation, not blocking the calibration (flash tier works fine)

---

## TL;DR — the ask

The local **sub2api** gateway (`weishaw/sub2api:latest`, `http://localhost:8080/antigravity`) serves Gemini.
- **Flash tier (`gemini-3-flash-agent`) sustains 12 concurrent workers cleanly** — 0 transport errors, 0 abstain.
- **Pro tier (`gemini-3.1-pro-low`) collapses under concurrency:** at **6 workers → 54% of calls fail** with `ConnectionResetError(104, 'Connection reset by peer')` / `RemoteDisconnected`; at **2 workers it's ~3.5%** (4/113) but still non-zero.

User's hypothesis (observed live): **both concurrent workers get routed to the *same* upstream Gemini account**, so they contend on one account's connection/quota and the upstream resets the peer. We want to confirm the routing behavior and find how to get clean concurrent pro throughput (or confirm pro is effectively serial and size the pipeline accordingly).

**Why it matters:** the production goal is to run a Gemini FP-review pass over **47k+ JHB Vexcel detector predictions**. Flash is fast but has a capability ceiling (confidently confuses gridded glass skylights with PV). Pro-low was expected to fix that.

> ⚠️ **PRIORITY REASSESSMENT (added after testing pro-low end-to-end):** pro-low **does NOT fix the skylight ceiling** — its residual TP-kills are still 100% "skylight" (re-kills 5 of flash's 7 with more skylight-specific reasoning), TP-keep 0.912 vs flash 0.926. The skylight↔PV confusion is **model-independent** (genuine overhead ambiguity / possible RA noise). Pro-low *is* a better overall judge (balanced 0.913 / κ 0.808 / FP-cut 0.913 vs flash 0.899/0.794/0.872 — **+4pp FP-cut**), so solving its throughput is still worth ~4pp more FP-cut at the same TP — but the original motivation (pro fixes skylights) is **gone**. Decide if +4pp FP-cut justifies the routing work; if not, this can be deprioritized and the 47k run goes on **flash multi-scale @ 12 workers** with the skylight band routed to human. At 2 workers, 47k chips on pro ≈ many hours regardless.

---

## Evidence (model-id × behavior, measured today)

All via the **native** path `POST /antigravity/v1beta/models/<MODEL>:generateContent` (the project's working path — see `[[project_gemini_sub2api]]`; OpenAI-compat `/v1/chat/completions` returns 401 "accounts exhausted" for this identity).

| model id | result |
|---|---|
| `gemini-3-flash-agent` | ✅ works, 12 workers clean (production flash model) |
| `gemini-3.1-pro-low` | ✅ works single/low-concurrency; **resets under concurrency** (6w→54% fail, 2w→3.5%) |
| `gemini-3.1-pro-high` | ❌ deprecated upstream ("Gemini 3 Pro is no longer available, switch to 3.1") — **do not use** |
| `gemini-3-pro-high` | ❌ returns the "no longer available" text as the body |
| `gemini-3.1-pro-high-agent` | ❌ `503 No available Gemini accounts` |
| `gemini-3.1-pro-low-agent`, `gemini-pro-agent`, `gemini-3.1-pro-agent` | ❌ `503 No available accounts` |

**Pattern to explain:** the **`-agent`** suffix (which flash REQUIRES — `gemini-3-flash-agent`) makes the **pro** ids 503, while the **plain** `gemini-3.1-pro-low` (no suffix) works. So flash and pro appear to route through *different account pools / upstream paths*. The `-agent` family probably routes to the **`cli-proxy-api`** container (Antigravity/Gemini-CLI OAuth pool, which has flash accounts but no pro accounts), while plain `gemini-3.1-pro-*` routes to sub2api's own pool.

Full error strings (workers=6 run): `ConnectionError: ('Connection aborted.', ConnectionResetError(104, 'Connection reset by peer'))` and `ConnectionError: ('Connection aborted.', RemoteDisconnected('Remote end closed connection without response'))`.

---

## System map

Containers (`docker ps`):
- **`sub2api`** = `weishaw/sub2api:latest`, the gateway on :8080. **This is the routing brain.**
- **`sub2api-postgres`** (postgres:18) — almost certainly holds the **account pool** (accounts are managed in DB/admin UI, NOT in `.env`).
- **`sub2api-redis`** (redis:8) — likely rate-limit / round-robin cursor / session state.
- **`cli-proxy-api`** = `eceasy/cli-proxy-api:latest` — separate CLI-OAuth proxy; suspected target of the `-agent` model ids.
- (`watchtower` is crash-looping — unrelated, but note it may auto-pull `:latest` and change `sub2api` under you.)

Deploy dir: **`/home/gaosh/projects/sub2api-deploy/`**
- `docker-compose.yml` — service/env wiring (sub2api at line 26).
- `.env` (chmod 600, secrets) — has `GEMINI_OAUTH_CLIENT_ID/SECRET/SCOPES`, **`GEMINI_QUOTA_POLICY`**. Redact when quoting.
- `data/config.yaml` — has **`default.user_concurrency = 5`**, `default.rate_multiplier = 1`. So the gateway's *per-user* concurrency cap is 5 — yet pro resets at 6 AND at 2, so the binding constraint is **upstream per-account**, not this cap.
- `redis_data/`, `postgres_data/`.

Upstream project: `weishaw/sub2api` (search the public repo for the account selector / load-balancer). Key questions for the source/docs:
- How does it pick an account per request — round-robin, least-loaded, sticky-by-model, sticky-by-session? Is the selection **per-model-family** (so all pro requests pin to the one pro-capable account)?
- Is there a **per-account max-concurrency / RPM** it's supposed to enforce (and is it enforcing it for pro)?
- Does `GEMINI_QUOTA_POLICY` gate pro differently?

---

## Investigation plan (suggested)

1. **Confirm the routing.** Tail `docker logs -f sub2api` while issuing pro requests; see which account id each request binds to. Issue 2 concurrent pro calls → are they the same account? (confirms user's hypothesis). Repeat for flash to contrast.
2. **Count pro-capable accounts.** Inspect the sub2api admin UI / postgres `accounts` table: how many accounts, which are marked pro-eligible / not-exhausted? If only **one** account can serve `gemini-3.1-pro-low`, concurrency >1 will always contend → pro is serial-by-construction until more pro accounts are added/logged-in.
3. **Map `-agent` vs plain routing.** Find why `*-agent` 503s for pro but is REQUIRED for flash. Likely two pools (cli-proxy-api vs sub2api-native). Determine which path has pro capacity.
4. **Find the concurrency knob.** Is there a per-account concurrency/serialization setting in sub2api config or DB? Can we set per-account max-concurrency=1 and let the gateway queue instead of resetting? Can `user_concurrency` be raised once multiple pro accounts exist?
5. **Decide the operating mode.** Either (a) add/log-in more pro accounts so round-robin gives real parallelism, or (b) accept pro is serial and run the 47k pass with `--workers 1-2` + generous `--retries`/backoff, sized accordingly.

**Cheap reproducer** (from `ZAsolar`, env active):
```bash
KEY=$(grep -E "^GEMINI_API_KEY=" /home/gaosh/projects/solar_backdating/.env.gemini.local | cut -d= -f2-)
# fire N concurrent identical pro calls, watch how many reset:
for i in $(seq 1 6); do
  curl -s -m 60 -H "x-goog-api-key: $KEY" -H "Content-Type: application/json" \
    "http://localhost:8080/antigravity/v1beta/models/gemini-3.1-pro-low:generateContent" \
    -d '{"contents":[{"parts":[{"text":"reply OK"}]}]}' -o /tmp/proresp_$i.json -w "$i:%{http_code} " &
done; wait; echo; grep -l . /tmp/proresp_*.json | head
```
(Note: a *minimal text* payload may 400 — the real scorer's `_call_gemini` builds a fuller request. To repro faithfully, use the scorer below with `--limit` and varying `--workers`.)

**Faithful reproducer** (the actual harness):
```bash
source scripts/activate_env.sh
for W in 1 2 4 6; do
  python scripts/analysis/gemini_fp_review.py \
    --chip-targets-csv data/analysis/gemini_review_calib/chips_jhb_z20/chip_targets.csv \
    --output /tmp/proprobe_w$W.jsonl --summary /dev/null \
    --model gemini-3.1-pro-low --workers $W --max-tokens 8192 --retries 0 --limit 20
  python3 -c "import json,collections;c=collections.Counter('ok' if json.loads(l).get('pv_present') is not None else 'fail' for l in open('/tmp/proprobe_w$W.jsonl'));print('W=$W',dict(c))"
done
```

---

## Relevant code in THIS repo (the caller side — probably fine, but context)

- **`scripts/analysis/gemini_fp_review.py`** — single-image FP scorer. Threaded `ThreadPoolExecutor(--workers)` + `RateLimiter(--qps)`. Each chip = one `_call_gemini`. Per-chip try/except → on transport failure records `gemini_error` (this is how we see the ConnectionResets). `--retries` re-issues immediately (no backoff — a backoff might help the reset case).
- **`scripts/analysis/gemini_fp_review_multiscale.py`** — two-image variant (sends `[tight, wide]`), same threading model.
- Both import transport from **`solar_backdating/scripts/validation/gemini_solar_image_review.py`**: `_call_gemini`, `GeminiClientConfig`, `RateLimiter`, `extract_json_object`, `load_env_file`, `env_value`. The HTTP call (requests/urllib, retry/backoff behavior, connection pooling) lives there — **if you want client-side mitigation (retry-with-backoff on ConnectionReset, connection: close, lower keep-alive), patch there.**
- Env: **`/home/gaosh/projects/solar_backdating/.env.gemini.local`** — `GOOGLE_GEMINI_BASE_URL=http://localhost:8080/antigravity`, `GEMINI_API_FORMAT=native`, `GEMINI_NATIVE_PATH=/v1beta`, `GEMINI_API_KEY=…`. (Note: `GEMINI_MODEL` in there is still set to the deprecated `gemini-3.1-pro-high` — cosmetic, the scorers pass `--model` explicitly.)

## Constraints / don't-break

- **Do not touch the running calibration data** under `data/analysis/gemini_review_calib/` (active comparison artifacts).
- The flash production path (`gemini-3-flash-agent`, 12 workers) **works — don't change its config** while debugging pro.
- This is config/infra investigation; **temporal/backdating boundary does not apply** (this is gateway plumbing, edit in `sub2api-deploy/` or patch the transport in `solar_backdating/scripts/validation/`).
- Secrets: `sub2api-deploy/.env` and `.env.gemini.local` are gitignored — keep keys/OAuth out of any committed notes.

## Background (so the agent has the full picture)

Goal = replace RA human review of 47k+ JHB Vexcel predictions with Gemini, objective **保TP最大砍FP**. Calibration on conf≥0.95 band (94 TP / 47 FP) found multi-scale two-image (20m tight + 48m wide) is the best **flash** config (TP-keep 0.926 / FP-cut 0.872), but its residual errors are 7 confident **gridded-glass-skylight→PV** confusions — a flash capability ceiling. **Pro-low is being tested to break that ceiling**, which is why pro throughput suddenly matters. Full calibration write-up: memory `project_gemini_fp_review_calibration`. sub2api native-path lore: memory `project_gemini_sub2api`.
