# Data Collection Milestone

Assets:
- `raw_episode_s3_20s.mp4`: sample episode pulled from stored dataset.
- `generated_episode_20s.mp4`: generated sample episode from pipeline validation.
- `ep0_summary.json`: frame/action summary for one sampled episode.
- `raw_preview_strip_ep0.png`: quick frame-strip preview from one episode (`t=0..180`).
- `data_collection_runs.json`: run-level counts (episodes/frames/shards).

Key point for presentation:
- We created three large raw datasets (v1, v2 diverse, v3 collision-heavy) with action + physics aligned per frame.
