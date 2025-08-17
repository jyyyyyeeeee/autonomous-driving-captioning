!pip -q install pandas nltk bert-score rouge-score
import pandas as pd
import numpy as np
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction
from nltk.translate.meteor_score import meteor_score
from nltk.tokenize import wordpunct_tokenize  # ✅ 추가: METEOR 토큰화용
from rouge_score import rouge_scorer
from bert_score import score as bertscore
import nltk

# (처음 1회만)
nltk.download('wordnet'); nltk.download('omw-1.4')

# ===== CSV 경로 =====
GT_CSV        = "/content/drive/MyDrive/2025urop-captioning/추론용 GT.csv"
BASELINE_CSV  = "/content/drive/MyDrive/2025urop-captioning/baseline_predictions.csv"
FINETUNED_CSV = "/content/drive/MyDrive/2025urop-captioning/finetuned_predictions.csv"

# ===== 로드 & 정리 =====
gt = pd.read_csv(GT_CSV).rename(columns={"caption":"gt"})
base = pd.read_csv(BASELINE_CSV).rename(columns={"caption":"baseline"})
fine = pd.read_csv(FINETUNED_CSV).rename(columns={"caption":"finetuned"})

for df in (gt, base, fine):
    df["image_id"] = df["image_id"].astype(str).str.strip()
    df[df.columns[1]] = df[df.columns[1]].astype(str).str.strip()

# 공통 image_id만 평가 (순서 무관)
df = gt.merge(base, on="image_id", how="inner").merge(fine, on="image_id", how="inner")

# 누락/중복 체크(선택)
missing_base = set(gt.image_id) - set(base.image_id)
missing_fine = set(gt.image_id) - set(fine.image_id)
if missing_base: print(f"[주의] 베이스라인 누락 {len(missing_base)}개")
if missing_fine: print(f"[주의] 파인튜닝 누락 {len(missing_fine)}개")
if df.image_id.duplicated().any(): print("[주의] 중복 image_id 존재")

# ===== 공통 준비 =====
refs = [[r.split()] for r in df["gt"].tolist()]  # 각 샘플당 1개 reference
preds_base = [s.split() for s in df["baseline"].tolist()]
preds_fine = [s.split() for s in df["finetuned"].tolist()]
cc = SmoothingFunction().method4

def bleu_suite(preds, refs):
    w = {
        "BLEU-1": (1.0, 0.0, 0.0, 0.0),
        "BLEU-2": (0.5, 0.5, 0.0, 0.0),
        "BLEU-3": (1/3, 1/3, 1/3, 0.0),
        "BLEU-4": (0.25, 0.25, 0.25, 0.25),
    }
    return {k: corpus_bleu(refs, preds, weights=v, smoothing_function=cc) for k, v in w.items()}

# ✅ 수정: METEOR는 토큰 리스트 필요
def meteor_avg(hyps_str, refs_str):
    scores = []
    for h, r in zip(hyps_str, refs_str):
        h_tok = wordpunct_tokenize(h.lower())
        r_tok = wordpunct_tokenize(r.lower())
        scores.append(meteor_score([r_tok], h_tok))
    return float(np.mean(scores))

def rougeL_avg(hyps, refs_str):
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    vals = [scorer.score(r, h)['rougeL'].fmeasure for r, h in zip(refs_str, hyps)]
    return float(np.mean(vals))

def bert_f1(hyps, refs_str):
    _, _, F = bertscore(hyps, refs_str, lang='en', verbose=False)
    return float(F.mean())

refs_str = df["gt"].tolist()
base_str = df["baseline"].tolist()
fine_str = df["finetuned"].tolist()

# ===== Baseline =====
b_bleu = bleu_suite(preds_base, refs)
b_meteor = meteor_avg(base_str, refs_str)
b_rougeL = rougeL_avg(base_str, refs_str)
b_bert = bert_f1(base_str, refs_str)

# ===== Finetuned =====
f_bleu = bleu_suite(preds_fine, refs)
f_meteor = meteor_avg(fine_str, refs_str)
f_rougeL = rougeL_avg(fine_str, refs_str)
f_bert = bert_f1(fine_str, refs_str)

# ===== 표로 출력 =====
results = pd.DataFrame({
    "Metric": ["BLEU-1","BLEU-2","BLEU-3","BLEU-4","METEOR","ROUGE-L","BERTScore-F1"],
    "Baseline": [b_bleu["BLEU-1"], b_bleu["BLEU-2"], b_bleu["BLEU-3"], b_bleu["BLEU-4"],
                 b_meteor, b_rougeL, b_bert],
    "Finetuned": [f_bleu["BLEU-1"], f_bleu["BLEU-2"], f_bleu["BLEU-3"], f_bleu["BLEU-4"],
                  f_meteor, f_rougeL, f_bert],
})
print(results.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# (선택) 저장
# results.to_csv("/content/drive/MyDrive/2025urop-captioning/eval_summary.csv", index=False)
