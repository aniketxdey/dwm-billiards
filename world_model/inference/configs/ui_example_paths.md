# UI Example Paths

Use these as a starting point in the UI fields (adjust to your current checkpoint):

- Train config:
  - `world_model/training/configs/dit_df_joint_v1v2v3_ctx8_2xh100_1521m_240m_hero.yaml`
- World model checkpoint (example checkpoint):
  - `/home/ubuntu/maat/world_model/training/runs/<run_id>/checkpoints/ckpt_072000000.pt`
- VAE checkpoint:
  - `/home/ubuntu/maat/vae_training/runs/vae_60m_1xa100_20260220_204310_run01/checkpoints/ckpt_060000000.pt`
- Eval shard manifest:
  - `/home/ubuntu/maat/world_model/training/manifests/joint_v1v2v3_full_400k/eval_shards.txt`
