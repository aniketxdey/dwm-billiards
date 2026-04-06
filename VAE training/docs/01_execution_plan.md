# VAE Execution Plan (1x A100 Fast Run)

## Goal
Get a high-signal VAE baseline quickly without sacrificing quality checks.

## Run spec
- Frame budget: `5,000,000`
- Review milestone: `1,000,000` (explicit milestone event + checkpoint)
- Source: full canonical dataset (`full_20260220_112101`)
- Sampling: randomized across shards/episodes/frames
- Checkpoints: every `100,000` frames
- Preview renders: every `10,000` frames

## Why this setup
- 1x A100 is sufficient for `72x128` VAE baseline.
- 1M-frame budget is enough to validate architecture and data pipeline quickly.
- Frequent previews/checkpoints allow early quality decisions.

## Phases
1. **Data staging**
   - Sync full shard set + metadata from S3 to local SSD.
2. **Frame cache build**
   - Build a local contiguous frame cache for high dataloader throughput.
3. **Training run**
   - Train Conv VAE with mixed precision and quality metrics.
4. **Review gates**
   - Primary review at `1,000,000` frames, then continue/adjust if quality is good.
5. **Decision after run**
   - Continue with larger frame budget or tune architecture/loss.

## Quality gates
- Reconstruction visuals improve checkpoint-over-checkpoint.
- LPIPS/PSNR trend in the right direction.
- KL value stable (no posterior collapse or exploding KL).
- No visual artifacts that break pool-table structure.

## Scale decision rule
- If quality and stability are good by `100k` and throughput is bottlenecked, scale up GPUs for larger run.
- If quality is weak, tune loss/model first before scaling compute.
