import json
import argparse
import sys
import re
import evaluate
import math

precision_metric = evaluate.load("precision")
recall_metric = evaluate.load("recall")
f1_metric = evaluate.load("f1")

def extract_stress_binary(stress_pattern: str):
    """
    將包含 <stress> 標籤的字串，轉換為 0/1 陣列與乾淨的單字列表。
    """
    # 1. 預處理防呆：確保標籤前後一定有空白，避免 LLM 生成時標籤與單字黏連
    safe_pattern = stress_pattern.replace("<stress>", " <stress> ").replace("</stress>", " </stress> ")
    tokens = safe_pattern.split()
    
    binary_array = []
    clean_words = []
    is_stressed = False  # 狀態開關
    punctuation = {",", ".", "?", "!", ";", ":"}
    
    # 2. 逐一掃描 Token
    for token in tokens:
        if token == "<stress>":
            is_stressed = True   # 開啟重音狀態
        elif token == "</stress>":
            is_stressed = False  # 關閉重音狀態
        elif token in punctuation:
            continue
        else:
            # 遇到一般單字，根據目前的狀態紀錄 0 或 1
            binary_array.append(1 if is_stressed else 0)
            clean_words.append(token)
            
    return binary_array, clean_words

def normalize_text(text):
    if not isinstance(text, str):
        return str(text)
    text = text.lower()
    # 移除所有標點符號，只保留字母與數字
    text = re.sub(r'[^\w\s]', '', text)
    return text.strip()

def normalize_tasks(tasks_list):
    if not isinstance(tasks_list, list):
        return []
    normalized_list = []
    for item in tasks_list:
        if not isinstance(item, dict):
            continue
        new_item = {}
        # if 'transcription' in item:
        #     new_item['transcription'] = normalize_text(item['transcription'])
        if 'stress_pattern' in item:
            new_item['stress_pattern'] = normalize_text(item['stress_pattern'])
        normalized_list.append(new_item)
    return normalized_list

def tokenize_for_wer(text):
    """為英文 WER 計算設計的分詞器 (直接以空格切分單詞)"""
    norm = normalize_text(text)
    return norm.split() if norm else []

def edit_distance(ref_tokens, hyp_tokens):
    """計算 Levenshtein Distance (用於 WER)"""
    n = len(ref_tokens)
    m = len(hyp_tokens)
    if n == 0: return m
    if m == 0: return n

    dp = list(range(m + 1))
    for i in range(1, n + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, m + 1):
            cur = dp[j]
            cost = 0 if ref_tokens[i - 1] == hyp_tokens[j - 1] else 1
            dp[j] = min(
                dp[j] + 1,      # deletion
                dp[j - 1] + 1,  # insertion
                prev + cost,    # substitution
            )
            prev = cur
    return dp[m]

def compute_prf_metrics(predictions, references, average="binary"):
    """
    Computes precision, recall, and F1 using Hugging Face's `evaluate`.
    Args:
        predictions (List[int]): Model's predicted labels.
        references  (List[int]): True labels.
        average     (str): "binary", "macro", "micro", or "weighted".
                          Use "binary" for two-class tasks.
    Returns:
        Dict[str, float]: e.g. {"precision": 0.8, "recall": 0.75, "f1": 0.77}
    """
    p = precision_metric.compute(predictions=predictions, references=references, average=average)["precision"]
    r = recall_metric.compute(predictions=predictions, references=references, average=average)["recall"]
    f = f1_metric.compute(predictions=predictions, references=references, average=average)["f1"]

    return {"precision": p, "recall": r, "f1": f}

def calculate_metrics(predict_file, ground_truth_file, target):
    with open(predict_file, 'r', encoding='utf-8') as f_pred:
        predict_lines = f_pred.readlines()
    with open(ground_truth_file, 'r', encoding='utf-8') as f_gt:
        ground_truth_lines = f_gt.readlines()

    if len(predict_lines) != len(ground_truth_lines):
        print(f"Error: line count mismatch. pred={len(predict_lines)} gt={len(ground_truth_lines)}", file=sys.stderr)
        sys.exit(1)

    total_count = len(predict_lines)
    # slot_tp = slot_fp = slot_fn = 0
    tp = fp = fn = 0
    wer_errors = 0
    wer_ref_len = 0
    precision = 0.0
    recall = 0.0
    f1_score = 0.0
    mismatch_count = 0
    g_tp = g_tn = g_fp = g_fn = 0
    gender_female = 0

    predictions = []
    references = []

    wer_errors_0 = 0
    wer_errors_1 = 0
    ############ 因應target 評估
    for i, (pred_line, gt_line) in enumerate(zip(predict_lines, ground_truth_lines), start=1):
        try:
            pred_data = json.loads(pred_line.strip())
            gt_data = json.loads(gt_line.strip())

            # predictions.jsonl
            # pred_tasks = pred_data.get("tasks", {})
            # gt_tasks = gt_data.get("tasks", {})

            pred_transcription = pred_data.get("pred_transcription", "")
            gt_transcription = gt_data.get("transcription", "")
            pred_transcription_0 = pred_data.get("pred_transcription_0", "")

            pred_stress = pred_data.get("pred_stress", "")
            gt_stress = gt_data.get("stress", "")

            pred_gender = pred_data.get("pred_gender", "")
            gt_gender = gt_data.get("gender", "")

            pred_stress_pattern,_ = extract_stress_binary(pred_stress)
            gt_stress_pattern,_ = extract_stress_binary(gt_stress)

            if len(pred_stress_pattern) != len(gt_stress_pattern):
                mismatch_count += 1
                print("Length mismatch", i)
                print(pred_transcription)
                print(pred_stress_pattern, len(pred_stress_pattern))
                print(gt_transcription)
                print(gt_stress_pattern, len(gt_stress_pattern))    
                continue
            else:
                for i, (gt, pred) in enumerate(zip(gt_stress_pattern, pred_stress_pattern)):
                    if gt == 1 and pred == 1:
                        tp += 1
                    elif gt == 1 and pred == 0:
                        fn += 1
                    elif gt == 0 and pred == 1:
                        fp += 1
                predictions.extend(pred_stress_pattern)
                references.extend(gt_stress_pattern)

                # --- 計算 WER (Word Error Rate) ---
                # 🛠️ 修正 2：強制清洗 pred_query，把 Qwen-Audio 的預設標籤殺掉
                raw_pred_transcription = pred_transcription
                clean_pred_transcription = re.sub(r'language\s+None|<asr_text>', '', raw_pred_transcription, flags=re.IGNORECASE).strip()
                clean_pred_transcription_0 = re.sub(r'language\s+None|<asr_text>', '', pred_transcription_0, flags=re.IGNORECASE).strip()

                query_ref_tokens = tokenize_for_wer(gt_transcription)
                query_hyp_tokens = tokenize_for_wer(clean_pred_transcription)
                query_hyp_tokens_0 = tokenize_for_wer(clean_pred_transcription_0)
                
                wer_errors += edit_distance(query_ref_tokens, query_hyp_tokens)
                wer_ref_len += len(query_ref_tokens)

                wer_errors_0 += edit_distance(query_ref_tokens, query_hyp_tokens_0)
                wer_errors_1 += edit_distance(query_hyp_tokens, query_hyp_tokens_0)

                if(target == "text_gts" or target == "text_gs"):
                    pred = gt = 0
                    if pred_gender == "female":
                        pred = 2
                    elif pred_gender == "male":
                        pred = 1

                    if gt_gender == "female":
                        gt = 2
                    elif gt_gender == "male":
                        gt = 1

                    if(gt == 2):
                        gender_female += 1
                    if gt == 2 and pred == 2:
                        g_tp += 1
                    elif gt == 2 and pred == 1:
                        g_fn += 1
                    elif gt == 1 and pred == 2:
                        g_fp += 1
                    elif gt == 1 and pred == 1:
                        g_tn +=1


            
        except Exception as e:
            print(f"Warning: failed at line {i}: {e}", file=sys.stderr)

    precision = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if fn == 0 else 0.0)
    recall = tp / (tp + fn) if (tp + fn) > 0 else (1.0 if fp == 0 else 0.0)
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    wer = wer_errors / wer_ref_len if wer_ref_len else 0.0
    wer_0 = wer_errors_0 / wer_ref_len if wer_ref_len else 0.0
    wer_1 = wer_errors_1 / wer_ref_len if wer_ref_len else 0.0

    metrics = compute_prf_metrics(predictions, references, average="binary")

    gender_accuracy = (g_tp + g_tn) / total_count if total_count > 0 else 0.0
    g_denom = math.sqrt((g_tp + g_fp) * (g_tp + g_fn) * (g_tn + g_fp) * (g_tn + g_fn))
    gender_mcc = (g_tp * g_tn - g_fp * g_fn) / g_denom if g_denom > 0 else 0.0
    gender_distribution = gender_female / total_count if total_count > 0 else 0.0

    return {
        "total_count": total_count,
        "mismatch_count": mismatch_count,
        "precision": precision,
        "recall": recall,
        "f1": f1_score,
        "query_wer_errors": wer_errors,
        "query_wer_errors_0": wer_errors_0,
        "query_wer_errors_1": wer_errors_1,
        "query_wer_ref_len": wer_ref_len,
        "query_wer": wer,
        "query_wer_0": wer_0,
        "query_wer_1": wer_1,
        "gender_acc": gender_accuracy,
        "gender_mcc": gender_mcc,
        "gender_distribution": gender_distribution,
        "metrics": metrics
    }

def main():
    parser = argparse.ArgumentParser(description="Calculate TinyStress Evaluation Metrics")
    parser.add_argument("predict_file")
    parser.add_argument("ground_truth_file")
    parser.add_argument("target")
    parser.add_argument("dataset")
    args = parser.parse_args()

    r = calculate_metrics(args.predict_file, args.ground_truth_file, args.target)
    print("-" * 60)
    print(f"{args.dataset} Evaluation Results")
    print("-" * 60)
    print(f"Total Samples: {r['total_count']}")
    print(f"Excluded (Length Mismatch):{r['mismatch_count']} ({(r['mismatch_count']/r['total_count'])*100:.2f}%)")
    print(f"ASR WER (Word Error Rate): {r['query_wer']:.4f} ({r['query_wer_errors']}/{r['query_wer_ref_len']})")
    print(f"WER (cleaned tagged transcript): {r['query_wer_0']:.4f} ({r['query_wer_errors_0']}/{r['query_wer_ref_len']})")
    print(f"WER (cleaned tagged transcript, ASR): {r['query_wer_1']:.4f} ({r['query_wer_errors_1']}/{r['query_wer_ref_len']})")
    print(f"Gender MCC:            {r['gender_mcc']:.4f}")
    print(f"Gender ACC:            {r['gender_acc']:.4f}")
    print(f"Gender distribution (female):            {r['gender_distribution']:.4f}")
    # print(f"Precision:            {r['precision']:.4f}")
    # print(f"Recall:               {r['recall']:.4f}")
    # print(f"F1 Score:             {r['f1']:.4f}")
    # print("right metrics")
    print(f"Precision:            {r['metrics']['precision']:.4f}")
    print(f"Recall:               {r['metrics']['recall']:.4f}")
    print(f"F1 Score:             {r['metrics']['f1']:.4f}")
    print("-" * 60)

if __name__ == "__main__":
    main()