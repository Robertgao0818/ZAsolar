# sync-collab Skill

Sync local ZAsolar repo to the collaboration repo (EddyZoulse01/SA_Solar) via git submodule.

## When to use

When the user wants to update the collaboration repo after local commits, or says "sync to collab", "push to Eddy's repo", "同步协作仓库".

## Architecture

- **Source repo**: `Robertgao0818/SA_Solar` (origin) — local project root
- **Collab repo**: `EddyZoulse01/SA_Solar` — has `ZAsolar/` as a git submodule pointing to source repo
- Submodule URL: `https://github.com/Robertgao0818/SA_Solar.git`

## Steps

### 1. Ensure local changes are committed and pushed to origin

```bash
git push origin main
```

### 2. Run the sync script

```bash
./scripts/sync_to_collab.sh
```

This script:
1. Pushes local main to `Robertgao0818/SA_Solar`
2. Clones `EddyZoulse01/SA_Solar` to a temp dir
3. Updates the ZAsolar submodule reference to the latest commit
4. Commits and pushes the updated submodule ref

### 3. Verify

Check https://github.com/EddyZoulse01/SA_Solar/tree/main/ZAsolar — it should show the submodule pointing to the latest commit hash.

## Manual sync (if script fails)

```bash
TMPDIR=$(mktemp -d)
git clone git@github.com:EddyZoulse01/SA_Solar.git "$TMPDIR"
cd "$TMPDIR"
git submodule update --init ZAsolar
cd ZAsolar && git pull origin main && cd ..
git add ZAsolar && git commit -m "Update ZAsolar submodule" && git push
rm -rf "$TMPDIR"
```

## Constraints

- SSH key must have write access to both repos
- Always push to origin first, then update collab submodule
- Never force-push to the collab repo without user confirmation
