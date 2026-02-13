import os
import json
import random
import cv2
import uuid
from tqdm import tqdm
from yt_dlp import YoutubeDL
from PIL import Image
from openai import OpenAI

# =========================
# 기본 설정
# =========================

DATA_ROOT = "playground/data"
FAKE_DIR = os.path.join(DATA_ROOT, "fake")
REAL_DIR = os.path.join(DATA_ROOT, "real")

TRAIN_JSON = os.path.join(DATA_ROOT, "train.json")
TEST_JSON = os.path.join(DATA_ROOT, "test.json")

TARGET_TRAIN = 10000
TARGET_TEST = 1000

FRAME_INTERVAL = 30  # 몇 프레임마다 추출할지

os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)

client = OpenAI()

# =========================
# 1. 유튜브 영상 다운로드
# =========================

def download_video(url, save_path):
    ydl_opts = {
        "outtmpl": save_path,
        "format": "mp4",
        "quiet": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# =========================
# 2. 프레임 추출
# =========================

def extract_frames(video_path, label_dir):
    cap = cv2.VideoCapture(video_path)
    count = 0
    saved = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if count % FRAME_INTERVAL == 0:
            img_name = f"{uuid.uuid4().hex}.png"
            img_path = os.path.join(label_dir, img_name)
            cv2.imwrite(img_path, frame)
            saved.append(img_path)

        count += 1

    cap.release()
    return saved

# =========================
# 3. GPT 레이블링 함수
# =========================

def generate_korean_explanation(image_path, label):
    """
    label: 0=fake, 1=real
    """

    if label == 0:
        instruction = """
이 이미지는 딥페이크 가능성이 있는 얼굴 이미지다.
왜 이 이미지가 fake인지 구체적으로 설명하라.
피부 질감, 눈, 입, 조명, 윤곽, 경계부 왜곡 등을 근거로 분석하라.
한국어로 상세히 설명하라.
"""
    else:
        instruction = """
이 이미지는 실제 인물의 자연스러운 얼굴 이미지다.
왜 이 이미지가 real인지 구체적으로 설명하라.
피부 질감, 눈, 입, 조명, 윤곽, 그림자 일관성 등을 근거로 분석하라.
한국어로 상세히 설명하라.
"""

    response = client.responses.create(
        model="gpt-5-mini",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": instruction},
                    {
                        "type": "input_image",
                        "image_url": f"file://{os.path.abspath(image_path)}",
                    },
                ],
            }
        ],
        max_output_tokens=500,
    )

    return response.output_text.strip()

# =========================
# 4. JSON 포맷 생성
# =========================

def build_sample(image_path, label):
    img = Image.open(image_path)
    width, height = img.size

    explanation = generate_korean_explanation(image_path, label)

    return {
        "image": image_path.replace(DATA_ROOT + "/", ""),
        "label": label,
        "cate": "deepfake" if label == 0 else "real",
        "width": width,
        "height": height,
        "conversations": [
            {
                "from": "human",
                "value": "<image>이 이미지는 real인가 fake인가?"
            },
            {
                "from": "gpt",
                "value": explanation
            }
        ]
    }

# =========================
# 5. 전체 파이프라인
# =========================

def main():
    with open("url.txt", "r") as f:
        urls = [u.strip() for u in f.readlines()]

    all_samples = []

    for url in tqdm(urls):
        video_id = uuid.uuid4().hex
        video_path = f"{video_id}.mp4"

        download_video(url, video_path)

        # --------------------------
        # ⚠️ 라벨링 전략
        # --------------------------
        # 예시: url에 'fake'가 포함되어 있으면 fake
        # 실제 운영 시 메타정보 기반으로 분류
        # --------------------------

        label = 0 if "fake" in url.lower() else 1
        label_dir = FAKE_DIR if label == 0 else REAL_DIR

        frames = extract_frames(video_path, label_dir)

        for img_path in frames:
            sample = build_sample(img_path, label)
            all_samples.append(sample)

        os.remove(video_path)

        if len(all_samples) >= TARGET_TRAIN + TARGET_TEST:
            break

    random.shuffle(all_samples)

    train_data = all_samples[:TARGET_TRAIN]
    test_data = all_samples[TARGET_TRAIN:TARGET_TRAIN + TARGET_TEST]

    with open(TRAIN_JSON, "w", encoding="utf-8") as f:
        json.dump(train_data, f, indent=4, ensure_ascii=False)

    with open(TEST_JSON, "w", encoding="utf-8") as f:
        json.dump(test_data, f, indent=4, ensure_ascii=False)

    print("Dataset generation complete.")

if __name__ == "__main__":
    main()
