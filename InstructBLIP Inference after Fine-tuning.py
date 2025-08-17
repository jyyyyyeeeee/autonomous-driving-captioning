!pip uninstall -y numpy torch torchvision torchaudio transformers accelerate peft bitsandbytes datasets xformers -q
!pip install -q "numpy==2.0.2"
!pip install -q --index-url https://download.pytorch.org/whl/cu121 \
  torch==2.3.1 torchvision==0.18.1 torchaudio==2.3.1
!pip install -q transformers==4.41.2 peft==0.11.1 datasets==2.19.2 tokenizers==0.19.1 bitsandbytes==0.43.1 accelerate==0.30.1
!pip install -q sentencepiece==0.1.99 timm==0.9.16

import torch
from transformers import AutoTokenizer, AutoImageProcessor, InstructBlipProcessor

MODEL_NAME = "Salesforce/instructblip-vicuna-7b"

# Vicuna/LLaMA 토크나이저(slow 권장)
txt_tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=False)

# Q-Former용 BERT 토크나이저
qf_tok  = AutoTokenizer.from_pretrained("bert-base-uncased")

# 이미지 프로세서
img_proc = AutoImageProcessor.from_pretrained(MODEL_NAME)

# 하나의 Processor로 조립
processor = InstructBlipProcessor(
    image_processor=img_proc,
    tokenizer=txt_tok,
    qformer_tokenizer=qf_tok,
)

tokenizer = processor.tokenizer
if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

print("OK. fast tokenizer?", getattr(tokenizer, "is_fast", False))


import os, glob
from peft import PeftModel
from transformers import InstructBlipForConditionalGeneration

bf16 = torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0] >= 8
dtype = torch.bfloat16 if bf16 else torch.float16

# 8bit 로드 (메모리 부족 시 load_in_4bit=True로 전환 가능)
base_model = InstructBlipForConditionalGeneration.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=dtype,
    load_in_8bit=True,
)

RUN_DIR = "/content/drive/MyDrive/instructblip_lora_run1"
ckpts = sorted(glob.glob(os.path.join(RUN_DIR, "checkpoint-*")), key=lambda p: int(p.split("-")[-1]))
assert ckpts, f"체크포인트가 없습니다: {RUN_DIR}"
adapter_path = ckpts[-1]
print("Using adapter:", adapter_path)

model = PeftModel.from_pretrained(base_model, adapter_path)
model.to("cuda")
model.eval()

from PIL import Image

IMAGE_PATH = "/content/drive/MyDrive/2025urop-captioning/추론용 이미지/d33f0c00-a05ea7e1.jpg"
question = "Describe the traffic scene for an autonomous driving agent."
prompt = f"USER: {question}\nASSISTANT: "

image = Image.open(IMAGE_PATH).convert("RGB")
enc = processor(images=image, text=prompt, return_tensors="pt",
                padding=True, truncation=True, max_length=512)
enc = {k: v.to(model.device) for k, v in enc.items()}

# 경고 방지
model.generation_config.max_length = None

with torch.inference_mode():
    out_ids = model.generate(
        **enc,
        do_sample=False,
        num_beams=3,
        max_new_tokens=100,
        no_repeat_ngram_size=4,
        repetition_penalty=1.15,
        early_stopping=True,
        length_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )

decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
answer = decoded.split("ASSISTANT:")[-1].strip()
print("\n=== PREDICTION ===\n", answer)


import os
import csv
from PIL import Image
from tqdm import tqdm

# ===== 경로 =====
IMAGE_DIR = "/content/drive/MyDrive/2025urop-captioning/추론용 이미지 100장"
OUT_CSV   = "/content/drive/MyDrive/2025urop-captioning/finetuned_predictions.csv"

question = "Describe the traffic scene for an autonomous driving agent."
prompt = f"USER: {question}\nASSISTANT: "

def is_image(fname):
    return fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp"))

files = [f for f in os.listdir(IMAGE_DIR) if is_image(f)]

with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "caption"])

    for fname in tqdm(files, desc="Finetuned captioning"):
        image_path = os.path.join(IMAGE_DIR, fname)
        image = Image.open(image_path).convert("RGB")

        enc = processor(images=image, text=prompt, return_tensors="pt",
                        padding=True, truncation=True, max_length=512)
        enc = {k: v.to(model.device) for k, v in enc.items()}

        # 경고 방지
        model.generation_config.max_length = None

        with torch.inference_mode():
            out_ids = model.generate(
                **enc,
                do_sample=False,
                num_beams=3,
                max_new_tokens=100,
                no_repeat_ngram_size=4,
                repetition_penalty=1.15,
                early_stopping=True,
                length_penalty=1.0,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
            )

        decoded = tokenizer.decode(out_ids[0], skip_special_tokens=True)
        caption = decoded.split("ASSISTANT:")[-1].strip()

        writer.writerow([fname, caption])

print(f"✅ 완료: {OUT_CSV}")



