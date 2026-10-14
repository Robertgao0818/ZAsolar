# Handoff — CT Census Overnight RunPod Run (community cloud)

**Date:** 2026-06-16
**Repo:** `/home/gaosh/projects/ZAsolar` (branch `main`)
**Goal:** Run the full Cape Town rooftop-solar census (2083 grid cells) on a
RunPod **community-cloud** pod, then back results up to Dropbox and stop the pod
once backup is verified.

---

## ⚠️ RUNTIME CORRECTIONS (2026-06-16) — read these FIRST; they override the original text below

Discovered while trying to start the run. Three of these will break the run if not applied.

### C0. The pod identity in the original handoff is STALE
- The handoff's pod `o2ccwdm1thuzxz` is **EXITED** (community pod, volume gone).
- The **actually-running pod** is `4vy7j4odp9aduy` (name `grand_gold_wallaby`,
  RTX 5090, **$0.34/hr**), created 2026-06-15 11:57. It happens to share the same
  direct-TCP endpoint `80.15.7.37:43783`.
- `.env` `RUNPOD_POD_ID=o2ccwdm1thuzxz` is therefore **wrong** — set it to
  `4vy7j4odp9aduy` (the lifecycle scripts use it). SSH host/port in `.env` are correct.
- RunPod account in use: `w.zou2@lse.ac.uk` (query via `RUNPOD_API_KEY` from `.env`:
  `RUNPOD_API_KEY=... runpodctl pod get 4vy7j4odp9aduy -o json`).

### C1. 🔴 SSH IS NOT WORKING — this is the current blocker (was never actually verified)
- The original "SSH wired (keyscan succeeded)" only proved the **port is open**, not auth.
- Direct SSH to `80.15.7.37:43783` **rejects every key tried** (`Permission denied (publickey)`):
  - `~/.ssh/id_ed25519` — even though it IS present in the pod's injected `PUBLIC_KEY` env.
  - `~/.runpod/ssh/RunPod-Key-Go` — the account key `runpodctl doctor` generated + synced.
  - Tried after a `runpodctl pod restart` too. Still rejected.
- ⇒ The pod's `/root/.ssh/authorized_keys` / sshd auth is broken at the infra layer,
  **not** a local-key problem. Nothing local left to tweak.
- **Fix tomorrow (pick one):**
  1. RunPod web console → Connect → **Web Terminal** (no SSH). Inspect
     `cat /root/.ssh/authorized_keys`, `service ssh status`; append the pubkey
     (`ssh-keygen -y -f ~/.ssh/id_ed25519`) manually and restart sshd.
  2. Simplest — **terminate this pod and create a fresh one** with the SSH public
     key registered in account settings *before* creation. Nothing is set up on the
     current pod yet (`/workspace` empty, no clone), so rebuild = zero loss.
- Note: tonight I ran `runpodctl doctor` (created local `~/.runpod/ssh/RunPod-Key-Go`
  + synced it to the account) and refreshed `.env` SSH host/port + `known_hosts`
  (values unchanged). All reversible.

### C2. 🔴 DISK: use `/workspace`, NOT `/root` — the scripts' `/root` defaults will fill the 30 GB container disk
- This pod = **30 GB container disk (`/root`)** + **300 GB volume at `/workspace`**.
  Tiles alone are 130–160 GB → they MUST go on `/workspace`.
- Clone the repos onto `/workspace` and override every heavy path. Export BEFORE
  running setup/run (the scripts are all env-overridable):
  ```bash
  export ZAS=/workspace/ZAsolar
  export CLS=/workspace/solar_cls
  export TILES_DISK=/workspace/tiles_disk
  export STATE=/workspace/census_state
  export LOGS=/workspace/census_logs
  export HF_DETECTOR_REPO=botao0818/zasolar-unified-reviewall-A   # public
  ```
- Results dir is computed by `get_results_root('ct', model_run='unifiedA_census_perdet')`
  → a **repo-relative** path, so with `ZAS=/workspace/ZAsolar` results also land on
  `/workspace`. Good. (Confirm the smoke test's printed "results dir resolves to:" is
  under `/workspace`.)
- `runpod_init.sh` defaults `WORKSPACE=/workspace/ZAsolar` already — consistent.

### C3. 🔴 venv vs system-python conflict — `activate_env.sh` will hard-fail on a runpod_init'd pod
- `ct_census_setup.sh` / `ct_census_run.sh` both `source scripts/activate_env.sh`,
  which **requires** a `$ZAS/.venv` and does `return 1` if missing.
- But `scripts/runpod_init.sh` installs deps into **system python**
  (`pip install --break-system-packages …`) and **never builds a `.venv`**.
- ⇒ As-is, the census scripts will abort at the first `activate_env.sh` line.
- **Fix (pick one), tomorrow:**
  - (a) Build a venv that inherits the base-image torch+cu128:
    `python3 -m venv --system-site-packages /workspace/ZAsolar/.venv` then
    `source scripts/activate_env.sh && pip install -r requirements.lock.txt`
    (lockfile is cu128-pinned → won't clobber torch). Do the same idea for `solar_cls`
    (its `activate_env.sh` sources this repo's). This is the path the scripts expect.
  - (b) OR strip the `activate_env.sh` calls from the two census scripts and run on
    system python (set `PYTHONPATH=$ZAS`, `PYTHONPATH=$CLS:$ZAS` for the cls step).
  - Prefer (a) — keeps the scripts unmodified and matches their stated assumption.

### C4. /dev/shm staging still correctly skipped
- `/workspace` here is a community **local** volume (fast NVMe), not a slow MFS network
  volume, so the runbook's "no `/dev/shm` staging" decision still holds. (If GPU util
  comes out <30% during the smoke test, revisit — stage per-batch tiles to `/dev/shm`,
  ~25 GB usable given 50 GB RAM.)

---

## State — what's already done

- **Plan + scripts written, committed, pushed to `main`** (pod gets them via
  `git clone`):
  - `76846d0` — pipeline scripts + `download_tiles.py` source-dir fix
  - `223cbfd` — baked in HF repo id default + solar_cls ssh clone url
- ~~**Pod is up & SSH wired:** `root@80.15.7.37:43783`~~ — **SUPERSEDED, see C0+C1.**
  The endpoint is open but **SSH auth is broken** (every key rejected); and the live
  pod is `4vy7j4odp9aduy`, not the `.env` pod id. Resolve SSH (C1) before anything else.
- **HF detector repo confirmed:** `botao0818/zasolar-unified-reviewall-A`
  (**public**; `best_model.pth` byte-identical to local — 176,243,385 B). Baked
  as the `HF_DETECTOR_REPO` default in `ct_census_setup.sh`. NOTE: HF username is
  `botao0818`, not the GitHub name `Robertgao0818`.

## Where everything lives (reference, do not re-derive)

- **SOP / rationale / disk estimate / monitoring design:**
  `docs/runbook/ct_census_overnight.md` (local-only — `docs/runbook/` is gitignored).
- **Scripts:** `scripts/ct_census_setup.sh` (Phase A env+ckpts+smoke),
  `scripts/ct_census_run.sh` (Phases B/C/D, tmux-hosted),
  `scripts/ct_census_merge.py`, `scripts/ct_census_status.sh` (monitor digest).
- **Detector source-dir fix:** `scripts/imagery/download_tiles.py` (writes tiles
  under `resolve_source_grid_id` dir with logical CPT filenames).
- **Durable decisions/context:** memory `project_ct_census_overnight_plan.md`
  (+ `project_ct_census_clsonly_baseline`, `project_ct_cpt_regrid`).

## Decisions LOCKED — do not re-litigate

- **Threshold = max recall + cls, no bespoke config.** Detector `unified_A` @
  `configs/postproc/v4_canonical.json` (no extra polygon-conf cut) + `solar_cls`
  dinov2 v2 `--classify-all` (aerial_2025 calibrated to 0.95 PV-recall).
  Confirmed on Li 16-grid to dominate the threshold-only frontier
  (area_F1 0.742 / σ_Bw 0.260 / cov50 0.829).
- **CPT addressing end-to-end** (geometry from `data/task_grid_cpt.gpkg`; no
  CPT↔G bookkeeping). Verified west/mid/east cells align.
- **Backup pod→Dropbox direct** via rclone `dropbox:` remote; **verify-then-stop**.
- **Disk: provision 250 GB** (42 tiles/grid ≈ 130–160 GB tiles; smoke test
  prints actual per-grid `du`).

## Next steps (the remaining work)

0. **PRECONDITION:** SSH must actually authenticate (see C1) — verify with
   `ssh root@<ip> -p <port> 'echo ok'` before proceeding.
1. **On the pod:** `git clone` ZAsolar **and** `solar_cls` as siblings **under
   `/workspace`** (see C2), run `scripts/runpod_init.sh`, then build the `.venv`
   (see C3) so `activate_env.sh` works. Export the C2 env block in every shell that
   runs the census scripts.
2. **Ship the cls checkpoint** (gitignored, 260 MB) P2P — NOT scp:
   - local: `runpodctl send /home/gaosh/projects/solar_cls/checkpoints/cls_pv_thermal_v2_dinov2_vits14/best_cls.pth`
   - local: scp its sibling `config.json` (≤100 MB ok) + `~/.config/rclone/rclone.conf`
   - pod: `runpodctl receive <code>` into `solar_cls/checkpoints/cls_pv_thermal_v2_dinov2_vits14/`
3. **Run setup:** `bash scripts/ct_census_setup.sh` (pulls detector from HF,
   builds glist, **smoke-tests one cell end-to-end** — validates chain + tile size).
4. **Launch:** `tmux new -s census 'bash scripts/ct_census_run.sh; bash'`.
5. **Monitor (context-economical):** main agent paces ~20–30 min wakes, each spawns
   a stateless sub-agent that SSHes and runs **only** `ct_census_status.sh`,
   returning a ≤6-line digest. Escalate on stale heartbeat / idle GPU / disk fill.
6. **Stop pod** only after `/root/census_state/BACKUP_VERIFIED` exists:
   `bash scripts/runpod_pod.sh stop`.

## Open questions / risks to resolve at runtime

- ~~HF repo public or private?~~ RESOLVED: `botao0818/zasolar-unified-reviewall-A`
  is public — no pod HF login needed.
- **solar_cls clone needs a github-authorized ssh key on the pod**
  (`CLS_GIT=git@github.com:Robertgao0818/solar_cls.git`). If absent, override
  `CLS_GIT` (https+token) or clone manually before setup.
- **WMS download throughput** (CT municipal WMS) is the likely Phase-B
  bottleneck; `DL_WORKERS=4` default. Smoke test gives a per-grid time signal.

## Suggested skills

- **runpod-ops** — pod lifecycle, env/venv/torch (Blackwell), file transfer
  (runpodctl/rclone/S3 rules), SSH/.env. Invoke for any pod-side action.
- **download-grids** — only if WMS download params need revisiting.
