# Project Status Update (March 4, 2026)

This document is a detailed checkpoint of the current state of the billiards world-model project, including what was implemented, what was evaluated, what is currently working, and what still needs work.

## 1) Scope of This Update

This update consolidates:

- final project writing artifacts (essay + slides + media),
- world-model experiment configs used for fairer DF vs baseline checks,
- distillation evaluation assets,
- interactive inference/game-route debugging and UI instrumentation,
- a distillation training stability fix in distributed logging/checkpoint flow.

## 2) What We Built So Far (End-to-End)

The project pipeline is now:

1. Simulate pool/billiards episodes and record trajectories/actions.
2. Train VAE and compress frames into latent shards.
3. Train action-conditioned DiT world models in latent space.
4. Train DF variants and compare rollout robustness at longer horizons.
5. Run rollout evaluations and export metrics/media.
6. Run interactive demo route with runtime action injection.
7. Prepare final paper/slides with quantitative + qualitative findings.

## 3) Key Research Findings to Date

### Main hypothesis

Diffusion Forcing (DF) improves long-horizon rollout physics consistency compared to one-step baseline training.

### Observed outcomes

- Long-horizon metrics favored DF strongly in historical and near-fair tests.
- Qualitative behavior:
  - wall bounces generally plausible,
  - collision continuity improved with DF + collision-heavy data,
  - remaining failure mode is identity consistency (color drift, occasional swallowing/merging).

### Distillation checkpoint status

Teacher vs student rollout evaluation artifact:

- `milestones_preview/final_project_media/distill_20260304/rollout_eval_summary.json`

Summary from that file:

- Teacher (`teacher_1521m_ckpt555m`) remains strong through horizon 32.
- Student (`student_573m_distill_30m`) is acceptable at short horizon but collapses at long horizon.

Implication: current student checkpoint is not presentation-grade for long-horizon realism; distillation requires additional stabilization.

## 4) Code Changes Included in This Update

### 4.1 Distillation trainer fix

File:

- `world_model_training/src/world_model_training/train_distill.py`

Change summary:

- Fixed logging/checkpoint flow to avoid distributed collective desync by ensuring reduced metrics are computed on all ranks when emitting logs.
- Simplified checkpoint trigger semantics with `did_ckpt = processed_samples >= next_ckpt` then conditional main-rank save.

Why this matters:

- Prevents failure mode where rank 0 and non-zero ranks diverge in collective usage around logging intervals.

### 4.2 Interactive game-route improvements

File:

- `world_model_inference/src/world_model_inference/game_route.py`

Implemented behavior:

- Two-click action capture:
  - click 1 starts capture and pauses generation,
  - click 2 commits action and resumes generation.
- Hard minimum action hold:
  - enforced minimum repeat frames for applied action (`HARD_MIN_ACTION_HOLD_FRAMES = 6`).
- Frame console and action HUD:
  - per-frame logs include source (`idle` vs `user_drag`), action magnitude, remaining repeats, latency.
- FPS control:
  - slower generation for debugging action response.
- Canvas interaction cleanup:
  - removed upload-style behavior from action canvas.
- Click mapping fix:
  - map display coordinates to model coordinates using actual displayed image size.
- Visual action overlays:
  - translucent red click-area markers to show action region,
  - short-lived visual for committed start/end drag points.

Operational status:

- Interactive routes are running remotely and exposed locally through SSH tunnels on:
  - `http://localhost:7864`
  - `http://localhost:7865`

## 5) New / Updated Config Assets

### Inference preview configs

- `world_model_inference/configs/preview_teacher_555m_20260304.yaml`
- `world_model_inference/configs/preview_student_distill_30m_20260304.yaml`

### Training/eval configs

- `world_model_training/configs/dit_fair_baseline_ctx8_2xh100_120m_resume60m.yaml`
- `world_model_training/configs/dit_fair_df_ctx8_2xh100_120m_resume60m.yaml`
- `world_model_training/configs/rollout_eval_joint_teacher555m_vs_distill573m_20260304.yaml`

Purpose:

- preserve exact experiment definitions used for fairer baseline-vs-DF discussion and teacher-vs-student evaluation.

## 6) Final Project Writing Assets

### Essay

- Source: `docs/final_project_essay/main.tex`
- PDF: `docs/final_project_essay/main.pdf`
- Bibliography: `docs/final_project_essay/refs.bib`
- Readme: `docs/final_project_essay/README.md`

### Slides

- Research deck source: `docs/final_project_slides/presentation_II_research.tex`
- Research deck PDF: `docs/final_project_slides/presentation_II_research.pdf`
- Basics deck source: `docs/final_project_slides/presentation_I_basics.tex`
- Basics deck PDF: `docs/final_project_slides/presentation_I_basics.pdf`
- Readme: `docs/final_project_slides/README.md`

### Media packaged for paper/deck/demo

- `docs/final_project_essay/assets/vae_latent_compare_ep0_13s.gif`
- `docs/final_project_essay/assets/latest_resume555m_clip_002.gif`
- `milestones_preview/final_project_media/distill_20260304/preview_teacher_clip_000.gif`
- `milestones_preview/final_project_media/distill_20260304/preview_student_clip_000.gif`
- `milestones_preview/final_project_media/distill_20260304/rollout_compare_clip_000.gif`

## 7) Reproduction Commands (Reference)

These are reference-style commands. Paths and compute environment must match your machine/cluster.

### Build essay PDF

```bash
cd docs/final_project_essay
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

### Build slide PDFs

```bash
cd docs/final_project_slides
pdflatex presentation_I_basics.tex
pdflatex presentation_I_basics.tex
pdflatex presentation_II_research.tex
pdflatex presentation_II_research.tex
```

### Run distillation training

```bash
python -m world_model_training.train_distill \
  --config world_model_training/configs/dit_distill_joint_v1v2v3_ctx8_2xh100_573m_from480m_60m.yaml
```

### Run teacher vs student rollout eval

```bash
python -m world_model_training.eval_rollout \
  --config world_model_training/configs/rollout_eval_joint_teacher555m_vs_distill573m_20260304.yaml
```

### Launch game-route interactive demo

```bash
python -m world_model_inference.game_route --host 0.0.0.0 --port 7864
python -m world_model_inference.game_route --host 0.0.0.0 --port 7865
```

## 8) Known Issues

- Distilled student quality is still unstable at long horizons.
- Interactive inference latency remains non-trivial for fully playable feel.
- Appearance identity consistency is still weaker than geometric/motion consistency.

## 9) Recommended Next Work (Ordered)

1. Stabilize distillation objective/schedule before replacing teacher in demo.
2. Add identity-consistency auxiliary loss (color/object persistence).
3. Run strict equal-budget ablations (same samples/seeded eval protocol).
4. Keep demo UX instrumentation (frame console/HUD) while tuning action response.

## 10) Notes for Presentation

For Presentation II (research/contribution), use:

- `docs/final_project_slides/presentation_II_research.pdf`

For the final essay, use:

- `docs/final_project_essay/main.pdf`

Both already reflect the current narrative:

- novelty = interactive physics consistency under action conditioning,
- finding = DF materially improves long-horizon robustness,
- limitations = identity drift/swallowing + latency + unfinished distillation.

