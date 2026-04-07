#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import json
import argparse
from typing import Any, Dict, List, Optional

import librosa
import torch
from qwen_asr import Qwen3ASRModel
from pathlib import Path

import matplotlib.pyplot as plt
# import seaborn as sns
import numpy as np


_CKPT_RE = re.compile(r"^checkpoint-(\d+)$")

# DEFAULT_PROMPT = """You are a professional Speech Analysis expert.
#     Your task is to analyze the provided audio and execute the task:
   
#     Transcription & Stress Detection (Stress Pattern): Transcribe the spoken English accurately. If a word is emphasized or stressed by the speaker, explicitly wrap it with <stress> and </stress> tags.
   
#     You must strictly follow this output format:
#     language English<asr_text>[Clean Transcription]<ssd>{"gender": "[Gender]", "stress_pattern": "[Tagged Transcription]"}
   
#     Example Outputs:
#     - single stress:
#     language English<asr_text>add seven hours to your two hour timer , right ?<ssd>{"stress_pattern": "add seven <stress> hours </stress> to your two hour timer , right ?"}
   
#     - no stress:
#     language English<asr_text>turn off the living room lights<ssd>{"stress_pattern": "turn off the living room lights"}
   
#     - multiple stresses:
#     language English<asr_text>I said red not blue<ssd>{"stress_pattern": "I said <stress> red </stress> not <stress> blue </stress>"}"""

def visualize_multi_stress(gen_out, stress_indices, audio_wav, audio_path, sr=16000, title="Multi-Stress Alignment Analysis"):
    """
    gen_out: model.generate 的回傳物件 (需含 cross_attentions)
    stress_indices: 包含所有 <stress> 位置的 list
    audio_wav: 原始音訊數值 (numpy array or tensor)
    """
    if not stress_indices:
        print("沒有找到 <stress> 標籤，跳過繪圖。")
        return

    num_stress = len(stress_indices)
    # 建立畫布：1 個波形圖 + num_stress 個注意力圖
    fig, axes = plt.subplots(num_stress + 1, 1, figsize=(15, 3 * (num_stress + 1)), sharex=True)
    
    # 取得音訊時間軸
    duration = len(audio_wav) / sr
    time_axis = np.linspace(0, duration, len(audio_wav))

    # --- 1. 繪製原始波形 ---
    axes[0].plot(time_axis, audio_wav, color='gray', alpha=0.4)
    axes[0].set_title("Original Audio Waveform")
    axes[0].set_ylabel("Amplitude")

    # --- 2. 遍歷每個 <stress> 標籤並繪圖 ---
    for i, token_idx in enumerate(stress_indices):
        ax = axes[i + 1]
        
        # 提取該 Token 的 Cross-Attention (取最後一層，平均所有 Head)
        # 結構: [step][layer][batch, head, query_len, key_len]
        # 我們取最後一層 [-1]，第 0 個 batch [0]，平均所有 head [mean(0)]
        # query_len 通常為 1 (因為是逐個生成的)，所以拿 [0, :]
        attn_weights = gen_out.cross_attentions[token_idx][-1][0].mean(dim=0)[0].cpu().numpy()
        
        # 將 Attention 幀數映射到時間軸
        attn_time = np.linspace(0, duration, len(attn_weights))
        
        # 繪製注意力曲線
        ax.fill_between(attn_time, attn_weights, color='orange', alpha=0.6, label=f'Stress Token @ Index {token_idx}')
        ax.set_ylabel("Attn Weight")
        ax.legend(loc='upper right')
        
        # 在波形圖上標註對應的高亮區 (找出注意力最高的地方)
        max_idx = np.argmax(attn_weights)
        peak_time = attn_time[max_idx]
        axes[0].axvline(x=peak_time, color='red', linestyle='--', alpha=0.5)
        axes[0].text(peak_time, ax.get_ylim()[1], f" S{i+1}", color='red', verticalalignment='bottom')

    axes[-1].set_xlabel("Time (seconds)")
    plt.suptitle(title, fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"plot_{audio_path}.png")
    plt.show()

# def plot_stress_attention(gen_out, token_index, audio_wav, sr=16000, id):
#     # 1. 取得特定 Token 的 Cross-Attention (假設取最後一層，並對 Head 取平均)
#     # gen_out.cross_attentions 結構: [step][layer][batch, head, query_len, key_len]
#     last_layer_attn = gen_out.cross_attentions[token_index][-1] 
#     attn_weights = last_layer_attn[0].mean(dim=0).cpu().numpy() # (1, audio_frames)
    
#     # 2. 建立畫布：上方波形，下方熱圖
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
#     # 上方：原始波形
#     time_axis = torch.linspace(0, len(audio_wav)/sr, len(audio_wav))
#     ax1.plot(time_axis, audio_wav, color='gray', alpha=0.5)
#     ax1.set_title("Audio Waveform")
    
#     # 下方：Attention 熱圖
#     sns.heatmap(attn_weights, ax=ax2, cmap="YlGnBu", cbar=False)
#     ax2.set_title(f"Cross-Attention for '<stress>' (Token Index: {token_index})")
    
#     plt.tight_layout()
#     plt.savefig(f"plot_{id}.png")
#     plt.show()

def find_latest_checkpoint(output_dir: str) -> Optional[str]:
    if not output_dir or not os.path.isdir(output_dir):
        return None

    best_step = None
    best_path = None
    for name in os.listdir(output_dir):
        m = _CKPT_RE.match(name)
        if not m:
            continue
        step = int(m.group(1))
        path = os.path.join(output_dir, name)
        if os.path.isdir(path) and (best_step is None or step > best_step):
            best_step = step
            best_path = path
    return best_path


def load_audio(path: str, sr: int = 16000):
    wav, _ = librosa.load(path, sr=sr, mono=True)
    return wav


def build_prefix_messages(prompt: str, audio_array=None):
    return [
        {"role": "system", "content": prompt or ""},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def build_prefix_text(processor, prompt: str) -> str:
    prefix_msgs = build_prefix_messages(prompt, None)
    prefix_text = processor.apply_chat_template(
        [prefix_msgs],
        add_generation_prompt=True,
        tokenize=False,
    )
    if isinstance(prefix_text, list):
        prefix_text = prefix_text[0]
    return prefix_text


def move_inputs_to_device(inputs: Dict[str, Any], device: str, model_dtype: torch.dtype):
    new_inputs = {}
    for k, v in inputs.items():
        if torch.is_tensor(v):
            v = v.to(device)
            if v.is_floating_point():
                v = v.to(model_dtype)
        new_inputs[k] = v
    return new_inputs


def batch_decode_text(processor, token_ids):
    if hasattr(processor, "batch_decode"):
        return processor.batch_decode(
            token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
    return processor.tokenizer.batch_decode(
        token_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )


def unwrap_generate_output(gen_out):
    if hasattr(gen_out, "sequences"):
        return gen_out.sequences
    if isinstance(gen_out, dict) and "sequences" in gen_out:
        return gen_out["sequences"]
    if isinstance(gen_out, (tuple, list)):
        return gen_out[0]
    return gen_out


def _extract_first_json_dict(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return {}
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:].strip()
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


# def try_parse_tasks_list(text: str) -> List[Dict[str, Any]]:
#     payload = _extract_first_json_array(text)
#     try:
#         obj = json.loads(payload)
#         if isinstance(obj, list):
#             return obj
#     except Exception:
#         pass
#     return []

def try_parse_tasks_dict(text: str) -> Dict[str, Any]:
    payload = _extract_first_json_dict(text)
    try:
        obj = json.loads(payload)
        # 🌟 修正：確保解析出來的是 dict 而不是 list
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return {}


def infer_one(
    asr_wrapper,
    audio_path: str,
    prompt: str = "",
    sr: int = 16000,
    max_new_tokens: int = 256,
    do_sample: bool = False,
    temperature: float = 1.0,
    top_p: float = 1.0,
) -> str:
    processor = asr_wrapper.processor
    model = asr_wrapper.model
    device = next(model.parameters()).device
    model_dtype = getattr(model, "dtype", torch.float16)

    wav = load_audio(audio_path, sr=sr)
    prefix_text = build_prefix_text(processor, prompt)

    inputs = processor(
        text=[prefix_text],
        audio=[wav],
        return_tensors="pt",
        padding=True,
        truncation=False,
    )

    prefix_len = int(inputs["attention_mask"][0].sum().item())
    inputs = move_inputs_to_device(inputs, device=device, model_dtype=model_dtype)

    gen_kwargs = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.inference_mode():
        # gen_out = model.generate(**inputs, **gen_kwargs)
        gen_out = model.generate(
            **inputs, 
            **gen_kwargs,
            return_dict_in_generate=True,
            output_attentions=True
        )

    # plot_stress_attention(gen_out, token_index, audio_wav, sr=16000, id)
    #############
    output_ids = unwrap_generate_output(gen_out)
    stress_token_id = processor.tokenizer.convert_tokens_to_ids("<stress>")
    visualize_multi_stress(gen_out, stress_token_id, wav, audio_path)
    #############
    if not torch.is_tensor(output_ids):
        raise TypeError(f"generate() returned unsupported type: {type(output_ids)}")

    if output_ids.dim() == 1:
        output_ids = output_ids.unsqueeze(0)

    if output_ids.size(1) > prefix_len:
        gen_only_ids = output_ids[:, prefix_len:]
    else:
        gen_only_ids = output_ids

    decoded = batch_decode_text(processor, gen_only_ids)[0].strip()
    return decoded


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line_id, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {line_id} in {path}: {e}")
    return data


def resolve_dtype(dtype_str: str, device: str) -> torch.dtype:
    if dtype_str == "bfloat16":
        return torch.bfloat16
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "float32":
        return torch.float32

    if device.startswith("cuda") and torch.cuda.is_available():
        try:
            major = torch.cuda.get_device_capability(device=device)[0]
        except Exception:
            major = torch.cuda.get_device_capability()[0]
        if major >= 8:
            return torch.bfloat16
        return torch.float16
    return torch.float32


def get_jsonl_name(input_jsonl: str) -> str:
    base = os.path.basename(input_jsonl)
    name, _ = os.path.splitext(base)
    return name


def write_ssd_prediction_jsonl(rows_out: List[Dict[str, Any]], output_root: str, jsonl_name: str):
    save_dir = os.path.join(output_root, jsonl_name)
    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, "predictions.jsonl")

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows_out:
            item = {
                "id": row["text_id"],
                "transcription": row.get("transcription", ""),
                "pred_transcription": row.get("pred_transcription", ""),
                "tasks": row.get("pred_tasks", {}),
            }
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[info] saved: {out_path}")


def parse_args():
    p = argparse.ArgumentParser("Qwen3-ASR SLU test script")

    p.add_argument("--exp_dir", type=str, required=True,
                   help="Experiment directory. Will load train_conf.json from this directory")
    p.add_argument("--auto_latest_checkpoint", action="store_true",
                   help="If exp_dir contains checkpoints, automatically use latest checkpoint")

    p.add_argument("--input_jsonl", type=str, required=True,
                   help="Input JSONL with fields like text_id, query, audio, prompt")

    p.add_argument("--output_root", type=str, default="checkpoints",
                   help='Root output dir. Default: "checkpoints"')

    p.add_argument("--device", type=str, default="cuda:0",
                   help='e.g. "cuda:0", "cuda:1", "cpu"')
    p.add_argument("--prompt-file", default="/datas/store162/annhung/Qwen3-SLU/prompt_ts.txt")
    p.add_argument("--target", default="text_ts")
    return p.parse_args()


def load_train_conf_from_exp_dir(exp_dir: str) -> Optional[List[Dict[str, Any]]]:
    if not exp_dir:
        return None

    train_conf_path = os.path.join(exp_dir, "train_conf.json")
    if not os.path.isfile(train_conf_path):
        raise FileNotFoundError(f"train_conf.json not found under exp_dir: {train_conf_path}")

    with open(train_conf_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, list) or len(cfg) != 2:
        raise ValueError("train_conf.json must be [training_args, model_args]")
    if not isinstance(cfg[0], dict) or not isinstance(cfg[1], dict):
        raise ValueError("Both train_conf entries must be dictionaries")
    return cfg


def main():
    args = parse_args()

    prompt = ""
    if args.prompt_file:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8").strip()

    train_conf = load_train_conf_from_exp_dir(args.exp_dir)
    if train_conf is None:
        raise ValueError("Unable to load train_conf from exp_dir")

    training_args_conf, model_args_conf = train_conf
    sr = int(training_args_conf.get("sr", 16000))
    max_new_tokens = int(training_args_conf.get("max_new_tokens", 256))
    do_sample = bool(training_args_conf.get("do_sample", False))
    temperature = float(training_args_conf.get("temperature", 1.0))
    top_p = float(training_args_conf.get("top_p", 1.0))
    dtype_str = str(model_args_conf.get("dtype", "auto"))

    model_path = args.exp_dir
    if args.auto_latest_checkpoint:
        latest_ckpt = find_latest_checkpoint(model_path)
        if latest_ckpt is None:
            raise ValueError(f"No checkpoint-* found under: {model_path}")
        model_path = latest_ckpt
        print(f"[info] use latest checkpoint: {model_path}")

    dtype = resolve_dtype(dtype_str, args.device)
    jsonl_name = get_jsonl_name(args.input_jsonl)

    asr_wrapper = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=dtype,
        device_map=args.device,
    )

    rows = load_jsonl(args.input_jsonl)
    rows_out = []

    for i, row in enumerate(rows, start=1):
        text_id = str(row.get("text_id", f"line{i}")).strip()
        audio_path = row.get("audio", "")
        # prompt = row.get("prompt", "")
        # print(prompt)
        transcription = row.get("transcription", "")

        if not audio_path:
            print(f"[skip] line {i}: no audio field")
            continue

        pred_raw = infer_one(
            asr_wrapper=asr_wrapper,
            audio_path=audio_path,
            prompt=prompt,
            sr=sr,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
        )

        if "<ssd>" in pred_raw:
            pred_transcription_raw = pred_raw.split("<ssd>")[0]
            pred_raw = pred_raw.split("<ssd>")[1]
            pred_transcription = pred_transcription_raw.replace("language English", "").replace("<asr_text>", "").strip()
        else:
            pred_transcription = ""

        print(f"pred_<ssd>{pred_raw}")
        rows_out.append({
            "text_id": text_id,
            "transcription": transcription,
            "pred_transcription": pred_transcription,
            "pred_raw": pred_raw,
            "pred_tasks": try_parse_tasks_dict(pred_raw),
        })

        print(f"[{i}/{len(rows)}] done: {text_id}")

    write_ssd_prediction_jsonl(rows_out=rows_out, output_root=args.output_root, jsonl_name=jsonl_name)


if __name__ == "__main__":
    main()
