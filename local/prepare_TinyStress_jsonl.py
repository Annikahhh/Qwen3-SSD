#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
import soundfile as sf
from pathlib import Path
from datasets import load_dataset, Audio
from tqdm import tqdm
import io
import soundfile as sf
import librosa
import numpy as np

# 全英文的系統提示詞 (SSD 任務)
# DEFAULT_PROMPT = """You are a professional Speech Analysis expert.
# Your task is to analyze the provided audio and execute the task:

# Transcription & Stress Detection (Stress Pattern): Transcribe the spoken English accurately. If a word is emphasized or stressed by the speaker, explicitly wrap it with <stress> and </stress> tags.

# You must strictly follow this output format:
# language English<asr_text>[Clean Transcription]<ssd>{"gender": "[Gender]", "stress_pattern": "[Tagged Transcription]"}

# Example Outputs:
# - single stress:
# language English<asr_text>add seven hours to your two hour timer , right ?<ssd>{"stress_pattern": "add seven <stress> hours </stress> to your two hour timer , right ?"}

# - no stress:
# language English<asr_text>turn off the living room lights<ssd>{"stress_pattern": "turn off the living room lights"}

# - multiple stresses:
# language English<asr_text>I said red not blue<ssd>{"stress_pattern": "I said <stress> red </stress> not <stress> blue </stress>"}
# """

def parse_args():
    p = argparse.ArgumentParser(description="Prepare TinyStress dataset for SFT")
    p.add_argument("--repo-id", required=True, help="HuggingFace repo path for TinyStress")
    p.add_argument("--download-dir", required=True)
    p.add_argument("--extract-root", required=True, help="Directory to save .wav files")
    p.add_argument("--jsonl-root", required=True, help="Directory to save processed .jsonl")
    p.add_argument("--splits", nargs="+", default=["train", "test"], choices=["train", "test", "validation"])
    p.add_argument("--prompt_file", default="/datas/store162/annhung/Qwen3-SLU/prompt/prompt_ts.txt")
    p.add_argument("--target", default="text_ts")
    return p.parse_args()

def main():
    args = parse_args()
    download_dir = Path(args.download_dir).resolve()
    extract_root = Path(args.extract_root).resolve()
    jsonl_root = Path(args.jsonl_root).resolve()
    
    prompt = ""
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    # 載入 Hugging Face 數據集
    print(f"[INFO] Loading dataset {args.repo_id} from HuggingFace...")
    ds = load_dataset(args.repo_id)
    ds = ds.cast_column("audio", Audio(decode=False))
    for split in args.splits:
        if split not in ds:
            print(f"[WARN] Split {split} not found in dataset. Skipping.")
            continue
            
        split_audio_dir = extract_root / split
        split_audio_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = jsonl_root / f"{split}.jsonl"
        out_path.parent.mkdir(parents=True, exist_ok=True)

        print(f"[INFO] Processing {split} split...")
        with out_path.open("w", encoding="utf-8") as f:
            for i, ex in enumerate(tqdm(ds[split])):
                # 1. 處理並保存音檔
                wav_filename = f"tiny_{split}_{i}.wav"
                wav_path = split_audio_dir / wav_filename
                
                # datasets 的 audio 欄位通常包含 'array' 和 'sampling_rate'
                # audio_data = ex['audio']['array']
                # sr = ex['audio']['sampling_rate']
                # sf.write(str(wav_path), audio_data, sr)

                # audio_bytes = ex['audio']
                # data, samplerate = sf.read(io.BytesIO(audio_bytes))
                # sf.write(str(wav_path), data, samplerate=16000)

                audio_dict = ex['audio']
                try:
                    # 安全地從字典中取出 bytes 或 path
                    if isinstance(audio_dict, dict) and audio_dict.get('bytes'):
                        data, orig_sr = sf.read(io.BytesIO(audio_dict['bytes']))
                    elif isinstance(audio_dict, dict) and audio_dict.get('path'):
                        data, orig_sr = sf.read(audio_dict['path'])
                    else:
                        continue
                        
                    # 真實的重採樣 (Resample) 到 16kHz
                    if orig_sr != 16000:
                        import librosa # 確保頂部有 import librosa
                        data = librosa.resample(data, orig_sr=orig_sr, target_sr=16000)
                    
                    # 正規化音量 (Scale to [-1, 1])
                    max_val = max(abs(data))
                    if max_val > 0:
                        data = data / max_val
                        
                    # 寫入正確的 16kHz 檔案
                    sf.write(str(wav_path), data, samplerate=16000)
                except Exception as e:
                    print(f"[WARN] 音檔處理失敗 {i}: {e}")
                    continue

                # 2. 準備語義結構 (根據 TinyStress 欄位調整)
                # query = ex.get("transcription", "")
                # 1. 先將原始字串拆解為單詞列表
                transcription = ex.get("transcription", "")
                words = ex.get("transcription", "").split() 
                # 2. 根據索引提取重音詞
                indices = set(ex.get("emphasis_indices", []))
                stressed_words = tagged_words = [
                    f"<stress> {word} </stress>" if i in indices else word 
                    for i, word in enumerate(words)
                ]
                stress_pattern = " ".join(stressed_words)

                tasks = {
                    "stress_pattern": stress_pattern
                }
                
                tasks_json = json.dumps(tasks, ensure_ascii=False)

                target_text = f"language English<asr_text>{transcription}<ssd>{tasks_json}"
                new_target_text_ts = f"language English<asr_text><ssd>{stress_pattern}"

                # gender
                gender_bi = ex.get("metadata", {}).get("gender", 0)
                gender = "female" if gender_bi == 2 else ("male" if gender_bi == 1 else "nan")
                tasks = {
                    "gender": gender,
                    "stress_pattern": stress_pattern
                }

                tasks_json = json.dumps(tasks, ensure_ascii=False)

                target_text_gts = f"language English<asr_text>{transcription}<ssd>{tasks_json}"
                new_target_text_gts = f"language English<asr_text><gender>{gender}<ssd>{stress_pattern}"
                new_target_text_tgs = f"language English<asr_text>{transcription}<gender>{gender}<ssd>{stress_pattern}"
                row = {
                    "text_id": f"tiny_{split}_{i}",
                    "audio": str(wav_path.resolve()),
                    "transcription": transcription,
                    "gender": gender,
                    "stress": stress_pattern,
                    "text_ts": target_text,
                    "text_gts": target_text_gts,
                    "text_s": new_target_text_ts,
                    "text_gs": new_target_text_gts,
                    "text_gts": new_target_text_tgs
                }

                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[INFO] Wrote {out_path}")

if __name__ == "__main__":
    main()