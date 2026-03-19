#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from huggingface_hub import hf_hub_download


def parse_args():
    p = argparse.ArgumentParser(
        description="Download Gatsby1984/MAC_SLU and prepare Kaldi data dirs"
    )
    p.add_argument("--repo-id", default="Gatsby1984/MAC_SLU")
    p.add_argument("--download-dir", required=True)
    p.add_argument("--extract-root", required=True)
    p.add_argument("--kaldi-root", required=True)
    p.add_argument(
        "--splits",
        nargs="+",
        default=["train", "dev", "test"],
        choices=["train", "dev", "test"],
    )
    return p.parse_args()


def ensure_local_file(repo_id: str, filename: str, local_path: Path) -> Path:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    if local_path.exists():
        return local_path
    cached_path = hf_hub_download(
        repo_id=repo_id,
        repo_type="dataset",
        filename=filename,
    )
    shutil.copy2(cached_path, local_path)
    return local_path


def download_split_files(repo_id: str, split: str, download_dir: Path):
    label_relpath = f"label/{split}_set.jsonl"
    audio_relpath = f"audio_{split}.tar.gz"

    label_local_path = download_dir / "label" / f"{split}_set.jsonl"
    audio_local_path = download_dir / f"audio_{split}.tar.gz"

    label_path = ensure_local_file(repo_id, label_relpath, label_local_path)
    audio_tar_path = ensure_local_file(repo_id, audio_relpath, audio_local_path)
    return label_path, audio_tar_path


def safe_extract_tar(tar_path: Path, out_dir: Path) -> None:
    marker = out_dir / ".extract_done"
    if marker.exists():
        return

    out_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        abs_out = out_dir.resolve()
        for member in tar.getmembers():
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(abs_out) + os.sep) and target != abs_out:
                raise RuntimeError(f"Unsafe path in tar file: {member.name}")
        tar.extractall(out_dir)

    marker.touch()


def load_jsonl(path: Path) -> List[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSONL at {path}:{i}") from e
    return rows


def build_wav_index(audio_root: Path) -> Dict[str, Path]:
    idx = {}
    for wav in audio_root.rglob("*.wav"):
        idx[wav.stem] = wav.resolve()
    return idx


def resolve_wav(raw_id: str, wav_index: Dict[str, Path]) -> Path:
    raw_id = str(raw_id).strip()
    return wav_index.get(raw_id, None)


def write_spk2utt(spk2utt_path: Path, utt2spk: Dict[str, str]) -> None:
    spk2utts = defaultdict(list)
    for utt, spk in utt2spk.items():
        spk2utts[spk].append(utt)

    with spk2utt_path.open("w", encoding="utf-8") as f:
        for spk in sorted(spk2utts):
            utts = sorted(spk2utts[spk])
            f.write(f"{spk} {' '.join(utts)}\n")


def main():
    args = parse_args()
    download_dir = Path(args.download_dir).resolve()
    extract_root = Path(args.extract_root).resolve()
    kaldi_root = Path(args.kaldi_root).resolve()

    for split in args.splits:
        label_path, audio_tar_path = download_split_files(
            args.repo_id, split, download_dir
        )
        split_audio_dir = extract_root / split
        safe_extract_tar(audio_tar_path, split_audio_dir)

        records = load_jsonl(label_path)
        wav_index = build_wav_index(split_audio_dir)
        if not wav_index:
            raise RuntimeError(f"No wav files found in {split_audio_dir}")

        out_dir = kaldi_root / split
        out_dir.mkdir(parents=True, exist_ok=True)

        wav_scp_path = out_dir / "wav.scp"
        text_path = out_dir / "text"
        utt2spk_path = out_dir / "utt2spk"
        spk2utt_path = out_dir / "spk2utt"

        missing = []
        utt2spk_map = {}

        with wav_scp_path.open("w", encoding="utf-8") as fwav, \
             text_path.open("w", encoding="utf-8") as ftext, \
             utt2spk_path.open("w", encoding="utf-8") as futt2spk:

            for i, r in enumerate(records, start=1):
                rid = f"id_{str(r.get('id', i))}"

                wav_path = resolve_wav(rid, wav_index)
                if wav_path is None:
                    print(f"ID {rid} NOT found")
                    missing.append(rid)
                    continue

                query = str(r.get("query", "")).strip()

                # 沒有 speaker 資訊時，用 uttid 當 spkid
                spk = rid

                fwav.write(f"{rid} {wav_path}\n")
                ftext.write(f"{rid} {query}\n")
                futt2spk.write(f"{rid} {spk}\n")
                utt2spk_map[rid] = spk

        write_spk2utt(spk2utt_path, utt2spk_map)

        if missing:
            miss_path = kaldi_root / f"{split}.missing_wavs.txt"
            miss_path.write_text("\n".join(missing) + "\n", encoding="utf-8")
            print(
                f"[WARN] {split}: skipped {len(missing)} records without exact wav match. "
                f"See {miss_path}"
            )

        print(f"[INFO] Wrote Kaldi data dir: {out_dir}")


if __name__ == "__main__":
    main()