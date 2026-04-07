#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm
import re

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
    p = argparse.ArgumentParser(description="Prepare Local Stress dataset for SFT")
    p.add_argument("--raw-dir", required=True, help="Directory containing the raw input .jsonl files")
    p.add_argument("--audio-dir", required=True, help="Directory containing the already saved .wav files")
    p.add_argument("--output-dir", required=True, help="Directory to save formatted .jsonl files")
    p.add_argument("--splits", nargs="+", default=["train", "test", "dev"], help="Dataset splits to process")
    p.add_argument("--prompt_file", default="/datas/store162/annhung/Qwen3-SLU/prompt/prompt_ts.txt")
    p.add_argument("--target", default="text_ts")
    return p.parse_args()

def fix_punctuation_spacing(text: str) -> str:
    """
    移除標點符號前多餘的空白。
    例如把 "timer , right ?" 變成 "timer, right?"
    """
    if not isinstance(text, str):
        return text
    # 尋找「一個以上的空白字元」後面跟著「標點符號 , . ? ! ; :」
    # 並將它們替換為「標點符號本身」
    return re.sub(r'\s+([,.\?!;:])', r'\1', text)

def main():
    args = parse_args()
    raw_dir = Path(args.raw_dir).resolve()
    audio_dir = Path(args.audio_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt = ""#DEFAULT_PROMPT
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    for split in args.splits:
        raw_jsonl_path = raw_dir / "gold_df.json" # 假設你的原始檔案叫 train.jsonl
        out_jsonl_path = output_dir / f"{split}.jsonl"

        if not raw_jsonl_path.exists():
            print(f"[WARN] 找不到原始資料檔: {raw_jsonl_path}，已跳過。")
            continue

        print(f"[INFO] Processing {split} split...")
        
        # 計算總行數供進度條使用
        with open(raw_jsonl_path, 'r', encoding='utf-8') as f:
            total_lines = sum(1 for _ in f)

        with open(raw_jsonl_path, 'r', encoding='utf-8') as fin, \
             open(out_jsonl_path, 'w', encoding='utf-8') as fout:
            
            missing_audio_count = 0
            
            for line in tqdm(fin, total=total_lines):
                line = line.strip()
                if not line:
                    continue
                    
                ex = json.loads(line)
                
                # 1. 解析新的資料格式
                wav_id = ex.get("id", "")
                
                # 假設你的音檔是直接放在 audio_dir 下，且檔名等於 id.wav
                # 如果你的音檔是依照資料夾分類的，這裡的路徑需要配合修改
                wav_path = audio_dir / f"{wav_id}.wav"
                
                # 🌟 核心需求：檢查音檔是否存在
                if not wav_path.exists():
                    missing_audio_count += 1
                    continue
                
                # 2. 處理文字與重音標籤
                # 原本是長字串，現在已經是 List: ["add", "seven", "hours"...]
                words = ex.get("src_sentence", [])
                
                # 將 List 組合成完整句子作為 ASR 的 ground truth
                # 如果標點符號需要特殊處理 (例如不想跟單字分開)，可以在這裡微調
                transcription = " ".join(words) 
                transcription = fix_punctuation_spacing(transcription)
                
                # 抓取重音單字 (使用新的 'gold_emphasis' 欄位，並加上防呆)
                indices = ex.get("gold_emphasis", [])
                
                stressed_words = tagged_words = [
                    f"<stress> {word} </stress>" if i in indices else word 
                    for i, word in enumerate(words)
                ]
                stress_pattern = " ".join(stressed_words)
                stress_pattern = fix_punctuation_spacing(stress_pattern)
                
                tasks = {
                    "stress_pattern": stress_pattern
                }
                tasks_json = json.dumps(tasks, ensure_ascii=False)

                target_text = f"language English<asr_text>{transcription}<ssd>{tasks_json}"
                new_target_text_ts = f"language English<asr_text><ssd>{stress_pattern}"

                # gender
                gender_num = int(wav_id[3])
                gender = "nan"
                if gender_num % 2 == 0:
                    gender = "female" 
                elif gender_num % 2 == 1:
                    gender = "male"

                tasks = {
                    "gender": gender,
                    "stress_pattern": stress_pattern
                }

                tasks_json = json.dumps(tasks, ensure_ascii=False)

                target_text_gts = f"language English<asr_text>{transcription}<ssd>{tasks_json}"
                new_target_text_gts = f"language English<asr_text><gender>{gender}<ssd>{stress_pattern}"
                # 4. 封裝並寫入
                row = {
                    "text_id": wav_id,
                    "audio": str(wav_path.resolve()),
                    "transcription": transcription,
                    "gender": gender,
                    "stress": stress_pattern,
                    "text_ts": target_text,
                    "text_gts": target_text_gts,
                    "text_s": new_target_text_ts,
                    "text_gs": new_target_text_gts
                }

                fout.write(json.dumps(row, ensure_ascii=False) + "\n")

        print(f"[INFO] {split} 處理完成。成功寫入: {out_jsonl_path}")
        if missing_audio_count > 0:
            print(f"[WARN] 共跳過了 {missing_audio_count} 筆找不到實體音檔的資料。")

if __name__ == "__main__":
    main()