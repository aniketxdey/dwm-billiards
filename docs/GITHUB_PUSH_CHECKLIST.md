# GitHub Push Checklist

Use this before every push.

## 1. Verify No Secrets Are Staged
```bash
git status --short
git diff --staged
```

Confirm these are not staged:
- `.env*` (except `.env.lambda.example`)
- `*.pem`, `*.key`
- raw dataset shards (`*.npz` from full data)
- local run outputs (`VAE training/runs/`, `wandb/`)

## 2. Verify No Oversized Files
GitHub hard limit is 100 MB per file.

```bash
find . -type f -size +95M -not -path "./.git/*"
```

If anything appears, keep it out of git (S3/local only).

## 3. Lint Core Documentation Links
Minimum docs to keep updated:
- `README.md`
- `docs/S3_ACCESS.md`
- `VAE training/docs/00_status_and_history.md`
- `VAE training/docs/03_runbook.md`

## 4. Commit
```bash
git add .
git commit -m "docs: update training and s3 workflow"
```

## 5. Push
```bash
git push origin <branch-name>
```

## Data Storage Rule
- GitHub: code, configs, docs, lightweight samples.
- S3: canonical dataset, checkpoints, previews, large artifacts.
