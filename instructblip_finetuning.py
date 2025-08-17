!pip uninstall -y numpy torch torchvision torchaudio transformers accelerate peft bitsandbytes datasets xformers -q
!pip install -q "numpy==2.0.2"
!pip install -q --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
!pip install -q transformers==4.41.2 peft==0.11.1 datasets==2.19.2 tokenizers==0.19.1 bitsandbytes==0.43.1 accelerate==0.30.1
import numpy as np, torch, transformers, peft, datasets, tokenizers
print("numpy", np.__version__)
print("torch", torch.__version__, "cuda?", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("peft", peft.__version__)
print("datasets", datasets.__version__)
print("tokenizers", tokenizers.__version__)
!pip install -q sentencepiece==0.1.99 timm==0.9.16
!rm -rf /root/.cache/huggingface/hub/models--Salesforce--instructblip-vicuna-7b

from transformers import AutoTokenizer, AutoImageProcessor, InstructBlipProcessor

MODEL_NAME = "Salesforce/instructblip-vicuna-7b"

# 1) Vicuna/LLaMA 토크나이저 (slow 권장)
txt_tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# 2) Q-Former용 BERT 토크나이저
qf_tok  = AutoTokenizer.from_pretrained("bert-base-uncased")

# 3) 이미지 프로세서
img_proc = AutoImageProcessor.from_pretrained(MODEL_NAME)

# 4) Processor 조립 (세 가지 모두 전달!)
processor = InstructBlipProcessor(
    image_processor=img_proc,
    tokenizer=txt_tok,
    qformer_tokenizer=qf_tok,
)

tokenizer = processor.tokenizer  # ↓ 전처리에서 쓰던 변수 유지
print("OK. fast?", getattr(tokenizer, "is_fast", False))

import torch, bitsandbytes as bnb
from transformers import InstructBlipForConditionalGeneration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

MODEL_NAME = "Salesforce/instructblip-vicuna-7b"
bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
dtype = torch.bfloat16 if bf16 else torch.float16

# 8bit 로딩 (OOM이면 load_in_4bit=True로 바꿔 QLoRA)
model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=dtype,
    load_in_8bit=True,
)

# (선택) 4bit일 때는 아래 한 줄도
# model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

# LoRA 타깃 모듈 (Vicuna 계열)
target_modules = ["q_proj","k_proj","v_proj","o_proj","gate_proj","down_proj","up_proj"]
lora_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=target_modules, bias="none", task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()
import os, random, pandas as pd, numpy as np
from datasets import Dataset, DatasetDict
from PIL import Image

# 경로/컬럼
IMG_DIR  = "/content/drive/MyDrive/2025urop-captioning/selected_3000"
CSV_PATH = "/content/drive/MyDrive/2025urop-captioning/captions_final_instruction2.csv"
COL_IMAGE, COL_INSTR, COL_ANS = "image_id", "text_input", "text_output"
VAL_RATIO, SEED = 0.1, 42

# CSV → split
df = pd.read_csv(CSV_PATH)
def to_path(x):
    p = str(x)
    return p if os.path.isabs(p) or os.path.exists(p) else os.path.join(IMG_DIR, p)
df["image_path"] = df[COL_IMAGE].apply(to_path)

random.seed(SEED); idx = list(range(len(df))); random.shuffle(idx)
cut = int(len(idx)*(1-VAL_RATIO))
train_df, val_df = df.iloc[idx[:cut]].reset_index(drop=True), df.iloc[idx[cut:]].reset_index(drop=True)
raw_ds = DatasetDict({"train": Dataset.from_pandas(train_df),
                      "validation": Dataset.from_pandas(val_df)})

# 프롬프트 → 라벨 마스킹
PROMPT_TPL = "USER: {q}\nASSISTANT: "

def build_example_np(path, q, a):
    prompt = PROMPT_TPL.format(q=str(q).strip())
    full_text = prompt + str(a).strip()

    tok_full   = tokenizer(full_text, truncation=True, max_length=512, add_special_tokens=True)
    tok_prompt = tokenizer(prompt,    truncation=True, max_length=512, add_special_tokens=True)
    labels = tok_full["input_ids"].copy()
    labels[:len(tok_prompt["input_ids"])] = [-100]*len(tok_prompt["input_ids"])

    with Image.open(path) as im:
        im = im.convert("RGB")
        proc = processor(images=im, text=full_text, return_tensors="np",
                         padding="max_length", max_length=512, truncation=True)

    L = proc["input_ids"].shape[1]
    return {
        "input_ids": np.array(proc["input_ids"][0], dtype=np.int64),
        "qformer_input_ids": np.array(proc["qformer_input_ids"][0], dtype=np.int64),  # ✅ 추가
        "attention_mask": np.array(proc["attention_mask"][0], dtype=np.int64),
        "pixel_values": np.array(proc["pixel_values"][0], dtype=np.float32),
        "labels": np.array(labels[:L], dtype=np.int64),
    }

def map_batch_np(batch):
    out = {k: [] for k in ["input_ids","qformer_input_ids","attention_mask","pixel_values","labels"]}
    for p, q, a in zip(batch["image_path"], batch[COL_INSTR], batch[COL_ANS]):
        ex = build_example_np(p, q, a)
        for k in out: out[k].append(ex[k])
    return out

proc_ds = raw_ds.map(
    map_batch_np, batched=True, num_proc=1, load_from_cache_file=False,
    remove_columns=raw_ds["train"].column_names, desc="preprocess"
)
proc_ds.set_format(type="torch", columns=["input_ids","qformer_input_ids","attention_mask","pixel_values","labels"])
# A→B 라이트 전환: 내부 변환 우회
proc_ds.reset_format()   # <- set_format 효과 해제

# 길이 안전 collator (qformer 포함)
import torch

class InstructBlipCollator:
    def __call__(self, features):
        def to_t(x, dt): return torch.as_tensor(x, dtype=dt)
        ids   = [to_t(f["input_ids"], torch.long) for f in features]
        qids  = [to_t(f["qformer_input_ids"], torch.long) for f in features]
        amask = [to_t(f["attention_mask"], torch.long) for f in features]
        labs  = [to_t(f["labels"], torch.long) for f in features]
        imgs  = [to_t(f["pixel_values"], torch.float32) for f in features]

        L_txt = max(x.size(0) for x in ids)
        L_qf  = max(x.size(0) for x in qids)
        PAD_TXT = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
        PAD_QF  = processor.qformer_tokenizer.pad_token_id or 0

        def pad(stack, L, pad_val):
            out=[]
            for a in stack:
                if a.size(0) < L:
                    out.append(torch.cat([a, torch.full((L-a.size(0),), pad_val, dtype=a.dtype)], dim=0))
                else:
                    out.append(a)
            return torch.stack(out)

        return {
            "input_ids": pad(ids, L_txt, PAD_TXT),
            "qformer_input_ids": pad(qids, L_qf, PAD_QF),
            "attention_mask": pad(amask, L_txt, 0),
            "labels": pad(labs, L_txt, -100),
            "pixel_values": torch.stack(imgs),
        }

data_collator = InstructBlipCollator()

from transformers import Trainer

class VLTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            qformer_input_ids=inputs["qformer_input_ids"],
            attention_mask=inputs["attention_mask"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

  from transformers import TrainingArguments, TrainerCallback
import torch, os, glob

bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8

# pad 토큰 보정(없으면 eos로)
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

# 콜백: 콘솔에 loss/eval_loss 강제 출력
class PrintLossCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs: return
        s = int(state.global_step)
        if "loss" in logs:
            print(f"step {s:>5} | loss {logs['loss']:.4f} | lr {logs.get('learning_rate', float('nan')):.2e}")
        if "eval_loss" in logs:
            print(f"[eval] step {s:>5} | eval_loss {logs['eval_loss']:.4f}")

args = TrainingArguments(
    output_dir="/content/instructblip_lora_run1",
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,

    logging_strategy="steps",
    logging_steps=20,                 # 20스텝마다 로그
    eval_strategy="steps",      # ✔ 4.41.2에서는 이 키 사용
    eval_steps=200,                   # 200스텝마다 검증
    save_strategy="steps",
    save_steps=200,                   # 200스텝마다 저장
    save_total_limit=3,

    dataloader_num_workers=4,
    bf16=bf16,
    fp16=not bf16,
    report_to="none",
    load_best_model_at_end=True,
    remove_unused_columns=False,
    save_safetensors=True,
    disable_tqdm=True,                # ✔ 진행바 끄고 콜백 로그가 잘 보이게
)

# ⛏️ PeftModel.forward가 inputs_embeds를 전달하지 않도록 패치
from types import MethodType
from peft import PeftModel

def _patched_forward(self,
                     input_ids=None,
                     attention_mask=None,
                     qformer_input_ids=None,
                     pixel_values=None,
                     labels=None,
                     **kwargs):
    # PEFT 내부 특수 키 제거
    if hasattr(self, "special_peft_forward_args"):
        kwargs = {k: v for k, v in kwargs.items() if k not in self.special_peft_forward_args}
    # ❗ 문제의 키 제거
    kwargs.pop("inputs_embeds", None)
    # BLIP은 qformer_input_ids를 명시적으로 받음
    return self.base_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        qformer_input_ids=qformer_input_ids,
        pixel_values=pixel_values,
        labels=labels,
        **kwargs,
    )

model.forward = MethodType(_patched_forward, model)
print("✅ Patched PeftModel.forward (no inputs_embeds)")

trainer = VLTrainer(
    model=model,
    args=args,
    train_dataset=proc_ds["train"],
    eval_dataset=proc_ds["validation"],
    data_collator=data_collator,
)

trainer.add_callback(PrintLossCallback())

# 최근 체크포인트 있으면 이어서
last = sorted(glob.glob(f"{args.output_dir}/checkpoint-*"), key=lambda p: int(p.split("-")[-1]))
resume = last[-1] if last else None
trainer.train(resume_from_checkpoint=resume)
!cp -r /content/instructblip_lora_run1 /content/drive/MyDrive/
