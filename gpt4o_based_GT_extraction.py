import os
import base64
import csv
from tqdm import tqdm
from openai import OpenAI

# 🔑 API 설정
client = OpenAI(api_key="본인 API 키 입력")

# 폴더 & CSV 경로
image_folder = "./images"   # 같은 프로젝트 폴더 안의 images 폴더
output_csv = "./output.csv"


# 📝 프롬프트
PROMPT = """
You are an expert in autonomous driving perception.

Given an image, describe the driving scene in clear, natural English sentences suitable for training autonomous vehicle captioning models.

Your description should include the following elements in natural language:
1. Number and type of visible vehicles (e.g., sedan, SUV), their **specific relative positions** (e.g., a silver sedan in the center lane, a blue SUV on the right at a distance), and whether traffic appears congested. Be **especially detailed and precise** when describing vehicles.
2. Number of pedestrians, their location (e.g., right sidewalk), and whether they are crossing or waiting.
3. State of the traffic lights (red, green, yellow) and traffic signs if any are visible (e.g., pedestrian crossing, speed limit).
4. The road structure, such as whether it is an intersection, roundabout, or straight road.
5. The surrounding environment, including building names with position if legible (e.g., “a building labeled 'Benjamin Cleaners' on the left”).
6. The weather (clear, rainy, etc.) and time of day (day/night).

Describe the scene as if you are helping an autonomous driving system understand the current environment.
Be accurate and structured. Avoid unnecessary details. Write short, factual sentences.

⚠️ Limit the total length of your response to **under 100 tokens**.
""".strip()


# 이미지 → Base64
def encode_image(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

# CSV 저장
with open(output_csv, mode="w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["image_id", "caption"])

    for image_file in tqdm(os.listdir(image_folder), desc="Processing images"):
        if not image_file.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
            continue

        image_path = os.path.join(image_folder, image_file)
        base64_image = encode_image(image_path)

        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": PROMPT},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                    ]
                }
            ],
            max_tokens=100
        )

        caption = response.choices[0].message.content.strip()
        writer.writerow([image_file, caption])

print(f"✅ 완료! 결과가 '{output_csv}'에 저장되었습니다.")


































