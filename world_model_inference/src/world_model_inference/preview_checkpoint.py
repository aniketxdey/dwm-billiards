from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import load_engine_from_preview_config, run_preview_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a rollout preview from a world-model checkpoint")
    parser.add_argument("--config", required=True, help="Path to YAML/JSON preview config")
    args = parser.parse_args()

    cfg_path = Path(args.config).expanduser().resolve()
    cfg, engine = load_engine_from_preview_config(cfg_path)
    artifacts = run_preview_from_config(cfg, engine)

    print(f"[preview] id={artifacts.preview_id}")
    print(f"[preview] out_dir={artifacts.output_dir}")
    print(f"[preview] summary={artifacts.summary_path}")
    if artifacts.video_paths:
        print(f"[preview] videos={len(artifacts.video_paths)} first={artifacts.video_paths[0]}")
    if artifacts.action_timeline_paths:
        print(
            f"[preview] action_timelines={len(artifacts.action_timeline_paths)} first={artifacts.action_timeline_paths[0]}"
        )


if __name__ == "__main__":
    main()
