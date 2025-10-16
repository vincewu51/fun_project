import os
import json
from pathlib import Path
import pyarrow.parquet as pq
import pyarrow as pa
import pandas as pd
import ffmpeg

def load_v3_metadata(root_v3):
    meta_dir = Path(root_v3) / "meta"
    info = json.load(open(meta_dir / "info.json"))
    episodes = [json.loads(line) for line in open(meta_dir / "episodes.jsonl")]
    return info, episodes

def split_parquet_into_episodes(root_v3, root_out, info, episodes):
    data_dir = Path(root_v3) / "data"
    out_data_dir = Path(root_out) / "data"
    out_data_dir.mkdir(parents=True, exist_ok=True)
    ep_frame_ranges = {}
    for chunk_folder in os.listdir(data_dir):
        chunk_path = data_dir / chunk_folder
        for pqf in os.listdir(chunk_path):
            if not pqf.endswith(".parquet"):
                continue
            df = pq.read_table(chunk_path / pqf).to_pandas()
            for ep in set(df["episode_index"].tolist()):
                sub = df[df["episode_index"] == ep].sort_values("frame_index")
                sub2 = sub.drop(columns=["episode_index"])
                tgt_folder = out_data_dir / "chunk-000"
                tgt_folder.mkdir(parents=True, exist_ok=True)
                pq.write_table(pa.Table.from_pandas(sub2), tgt_folder / f"episode_{ep:06d}.parquet")
                ep_frame_ranges[ep] = (sub["frame_index"].min(), sub["frame_index"].max())
    return ep_frame_ranges

def slice_video_for_episode(root_v3, root_out, ep_frame_ranges, info, camera_key="main"):
    video_dir = Path(root_v3) / "videos" / camera_key
    out_video_dir = Path(root_out) / "videos" / "chunk-000" / camera_key
    out_video_dir.mkdir(parents=True, exist_ok=True)
    fps = info.get("fps", 30)
    vid_src = next(video_dir.glob("*.mp4"))
    for ep, (fstart, fend) in ep_frame_ranges.items():
        t_start = fstart / fps
        t_end = (fend + 1) / fps
        out_vid = out_video_dir / f"episode_{ep:06d}.mp4"
        (ffmpeg.input(str(vid_src)).trim(start=t_start, end=t_end).setpts("PTS-STARTPTS")
         .output(str(out_vid), vcodec="copy", acodec="copy").run(overwrite_output=True))

def write_v2_metadata(root_out, info_v3, episodes, ep_frame_ranges):
    meta_dir = Path(root_out) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    info2 = {
        "codebase_version": "v2.0",
        "robot_type": info_v3.get("robot_type", ""),
        "total_episodes": len(episodes),
        "total_frames": sum(r[1] - r[0] + 1 for r in ep_frame_ranges.values()),
        "fps": info_v3.get("fps", 30),
        "splits": info_v3.get("splits", {}),
        "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
        "video_path": "videos/{video_key}/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.mp4"
    }
    with open(meta_dir / "info.json", "w") as f:
        json.dump(info2, f, indent=2)
    with open(meta_dir / "episodes.jsonl", "w") as fe:
        for ep in episodes:
            ep_meta = {"episode_index": ep["episode_index"]}
            fe.write(json.dumps(ep_meta) + "\n")
    with open(meta_dir / "episodes_stats.jsonl", "w") as fsj:
        for ep in episodes:
            idx = ep["episode_index"]
            fsj.write(json.dumps({
                "episode_index": idx,
                "length": ep_frame_ranges[idx][1] - ep_frame_ranges[idx][0] + 1
            }) + "\n")

def convert_v3_to_v2(root_v3, root_out):
    os.makedirs(root_out, exist_ok=True)
    info_v3, episodes = load_v3_metadata(root_v3)
    ep_ranges = split_parquet_into_episodes(root_v3, root_out, info_v3, episodes)
    for cam in info_v3.get("video_keys", ["main"]):
        slice_video_for_episode(root_v3, root_out, ep_ranges, info_v3, cam)
    write_v2_metadata(root_out, info_v3, episodes, ep_ranges)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-v3", type=str, required=True)
    parser.add_argument("--root-out", type=str, required=True)
    args = parser.parse_args()
    convert_v3_to_v2(args.root_v3, args.root_out)
