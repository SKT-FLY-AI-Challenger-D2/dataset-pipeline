import os
import json
import cv2
import uuid
import base64
import numpy as np
import concurrent.futures
import hashlib
import argparse
from tqdm import tqdm
from yt_dlp import YoutubeDL
from PIL import Image
from openai import OpenAI
from google import genai
from google.genai import types
from anthropic import Anthropic
from dotenv import load_dotenv
import sys 

load_dotenv()

# =========================
# 기본 설정
# =========================

DATA_ROOT = "playground/data"
FAKE_DIR = os.path.join(DATA_ROOT, "fake")
REAL_DIR = os.path.join(DATA_ROOT, "real")

# JSONL(append)로 저장: 중간에 꺼져도 누적 보존
TRAIN_JSONL = os.path.join(DATA_ROOT, "train.jsonl")
TEST_JSONL = os.path.join(DATA_ROOT, "test.jsonl")

# 진행상태 체크포인트
PROGRESS_JSON = os.path.join(DATA_ROOT, "progress.json")

# 테스트를 위해 1개 영상만 처리하도록 설정 (필요시 수정)
TEST_ONLY_ONE_VIDEO = False

NUM_FRAMES_PER_VIDEO = 20  # 영상 하나당 추출할 프레임 수

os.makedirs(FAKE_DIR, exist_ok=True)
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(DATA_ROOT, exist_ok=True)

# API 클라이언트 초기화
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
gemini_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))
anthropic_client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

# =========================
# 유틸리티 함수
# =========================

def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def append_jsonl(path: str, obj: dict) -> None:
    """한 줄에 한 JSON 객체를 append. 크래시 복원에 가장 안전."""
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def load_progress() -> dict:
    if os.path.exists(PROGRESS_JSON):
        try:
            with open(PROGRESS_JSON, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and "done_urls" in data and isinstance(data["done_urls"], list):
                return data
        except Exception:
            pass
    return {"done_urls": []}

def save_progress(progress: dict) -> None:
    """atomic write로 progress.json 깨짐 방지"""
    tmp = PROGRESS_JSON + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    os.replace(tmp, PROGRESS_JSON)

def url_to_key(url: str) -> str:
    """URL을 짧은 고정 키로 변환 (파일명/식별자용)"""
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:16]

def remove_samples_for_url(jsonl_path: str, target_url: str) -> int:
    """jsonl에서 특정 URL로 생성된 샘플만 제거"""
    if not os.path.exists(jsonl_path):
        return 0
    tmp = jsonl_path + ".tmp"
    removed = 0
    with open(jsonl_path, "r", encoding="utf-8") as fin, open(tmp, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                # 깨진 라인은 버림
                continue
            if obj.get("source_url") == target_url:
                removed += 1
                continue
            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    os.replace(tmp, jsonl_path)
    return removed

def collect_image_paths_for_url(train_jsonl: str, test_jsonl: str, target_url: str) -> list[str]:
    """특정 URL 샘플들이 참조하는 이미지 파일 경로들을 수집 (삭제용)"""
    paths = []
    for p in [train_jsonl, test_jsonl]:
        if not os.path.exists(p):
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("source_url") == target_url:
                    rel = obj.get("image")
                    if rel:
                        paths.append(os.path.join(DATA_ROOT, rel))
    # 중복 제거
    return sorted(set(paths))

def delete_images(paths: list[str]) -> int:
    deleted = 0
    for p in paths:
        if os.path.exists(p):
            try:
                os.remove(p)
                deleted += 1
            except Exception:
                pass
    return deleted

def reset_url_state(target_url: str) -> None:
    """특정 URL의 데이터/진행상태를 제거해서 그 URL부터 다시 돌릴 수 있게 함"""
    print(f"\n[RESET] Target URL: {target_url}")

    # 1) progress.json에서 done 제거
    progress = load_progress()
    before = len(progress.get("done_urls", []))
    progress["done_urls"] = [u for u in progress.get("done_urls", []) if u != target_url]
    after = len(progress["done_urls"])
    save_progress(progress)
    print(f"[RESET] progress.json updated: done_urls {before} -> {after}")

    # 2) JSONL에서 해당 URL 샘플이 참조하는 이미지 먼저 삭제
    img_paths = collect_image_paths_for_url(TRAIN_JSONL, TEST_JSONL, target_url)
    deleted_imgs = delete_images(img_paths)
    print(f"[RESET] deleted images: {deleted_imgs}")

    # 3) JSONL에서 해당 URL 샘플 제거
    removed_train = remove_samples_for_url(TRAIN_JSONL, target_url)
    removed_test = remove_samples_for_url(TEST_JSONL, target_url)
    print(f"[RESET] removed samples: train={removed_train}, test={removed_test}")

# =========================
# 1. 유튜브 영상 다운로드
# =========================

def download_video(url: str, save_path: str) -> None:
    ydl_opts = {
        "outtmpl": save_path,
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "quiet": True,
        "no_warnings": True,
    }
    with YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])

# =========================
# 2. 프레임 추출 (고정된 개수)
# =========================

def extract_frames(video_path: str, label_dir: str, video_key: str, num_frames: int = 20) -> list[str]:
    """
    ✅ 프레임 파일명을 결정적으로 생성:
    {video_key}_{frameIdx}.png
    -> 재실행해도 같은 파일명으로 저장되어 추적/리셋이 쉬움
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        print(f"Error: Video {video_path} has no frames.")
        cap.release()
        return []

    frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

    saved = []
    for i, idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ret, frame = cap.read()
        if not ret:
            continue

        img_name = f"{video_key}_{i:03d}.png"
        img_path = os.path.join(label_dir, img_name)
        cv2.imwrite(img_path, frame)
        saved.append(img_path)

    cap.release()
    return saved

# =========================
# 3. 개별 LLM 분석 함수
# =========================

def _base_prompt(label: int) -> str:
    return (
        f"Analyze this image. It is labeled as {'fake (deepfake)' if label == 0 else 'real'}. "
        f"Explain why it is {'fake' if label == 0 else 'real'} based on lighting, texture, shadows, and consistency. "
        "Keep the response under 300 chars."
    )

def get_openai_analysis(image_path: str, label: int) -> str:
    base64_image = encode_image_to_base64(image_path)
    prompt = _base_prompt(label)

    response = openai_client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": f"data:image/png;base64,{base64_image}"},
                ],
            }
        ],
    )
    return response.output_text

def get_gemini_analysis(image_path: str, label: int) -> str:
    prompt = _base_prompt(label)
    with open(image_path, "rb") as f:
        img_data = f.read()

    response = gemini_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[prompt, types.Part.from_bytes(data=img_data, mime_type="image/png")],
    )
    return response.text

MAX_IMAGE_BYTES = 5 * 1024 * 1024  # 5MB


def get_anthropic_analysis(image_path: str, label: int) -> str | None:
    # 1) 인코딩(네가 실제로 요청에 넣는 데이터)
    base64_image = encode_image_to_base64(image_path)

    # 2) Anthropic 기준으로 "실제 이미지 바이트" 체크 (중요!)
    try:
        image_bytes = base64.b64decode(base64_image, validate=True)
    except Exception as e:
        print(f"[SKIP] Anthropic invalid base64 for {image_path}: {e}")
        return None

    if len(image_bytes) > MAX_IMAGE_BYTES:
        print(f"[SKIP] Anthropic image too large AFTER encoding: {len(image_bytes)} bytes")
        return None

    prompt = _base_prompt(label)

    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": base64_image,
                            },
                        },
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        return response.content[0].text
    except Exception as e:
        # 혹시라도 남는 케이스는 안전하게 스킵
        if "image exceeds 5 MB maximum" in str(e):
            print("[SKIP] Anthropic rejected image > 5MB")
            return None
        print(f"[SKIP] Anthropic error: {e}")
        return None

# =========================
# 4. 결과 병합 (Expert Merger LLM)
# =========================

def merge_responses(responses: list[str], label: int) -> str:
    merger_prompt = f"""
You are an expert in AI-generated content analysis. Merge the following three model responses into
one unified answer. All responses agree on the image’s authenticity ({'fake' if label==0 else 'real'}). Prioritize explanations
mentioned by at least two models and omit points unique to a single model. Keep the response under 200 chars.
Follow these steps:
• Extract Common Ground: Identify overlapping details across all three responses.
• Filter Minority Claims: Discard observations mentioned by only one model unless critical.
• Structure Hierarchically: Group explanations by category for clarity.
• Maintain Original Format: Begin with “This is a {'fake' if label==0 else 'real'} image.” then semicolon-separated evidence.
• Avoid Redundancy: Rephrase overlapping points.
• Ensure Logical Consistency: Discard nonsensical/contradictory reasoning.

Model Responses:
1. {responses[0]}
2. {responses[1]}
3. {responses[2]}
"""
    response = openai_client.responses.create(
        model="gpt-5-mini-2025-08-07",
        input=[{"role": "user", "content": [{"type": "input_text", "text": merger_prompt}]}],
    )
    return response.output_text.strip()

# =========================
# 5. JSON 포맷 생성
# =========================

def build_sample(image_path: str, label: int, source_url: str, video_key: str) -> dict:
    # 🔹 이미지 사이즈 읽기
    try:
        img = Image.open(image_path)
        width, height = img.size
    except Exception as e:
        print(f"Error opening image {image_path}: {e}")
        width, height = 0, 0
    
    print(f"Analyzing image: {os.path.basename(image_path)} (Parallel)...")

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_openai = executor.submit(get_openai_analysis, image_path, label)
        future_gemini = executor.submit(get_gemini_analysis, image_path, label)
        future_anthropic = executor.submit(get_anthropic_analysis, image_path, label)

        try:
            openai_resp = future_openai.result()
            gemini_resp = future_gemini.result()
            anthropic_resp = future_anthropic.result()
        except Exception as e:
            # ✅ 즉시 중단: 원인 로그 남기고 예외 전파
            print(f"\n[FATAL] API error during multi-LLM analysis: {e}", file=sys.stderr)
            raise  # <- 중요: 여기서 멈춤

    unified_explanation = merge_responses([openai_resp, gemini_resp, anthropic_resp], label)

    rel_image_path = os.path.relpath(image_path, DATA_ROOT)

    return {
        "image": rel_image_path,
        "label": label,
        "cate": "deepfake" if label == 0 else "real",
        "width": width,
        "height": height,
        "source_url": source_url,
        "video_key": video_key,
        "conversations": [
            {"from": "human", "value": "<image> Is this image real or fake?"},
            {"from": "gpt", "value": unified_explanation},
        ],
    }

# =========================
# 6. url.txt 파싱 로직
# =========================

def parse_url_txt(file_path: str) -> list[dict]:
    videos = []
    current_label = None

    if not os.path.exists(file_path):
        return videos

    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith("fake"):
                current_label = 0
                continue
            elif line.startswith("real"):
                current_label = 1
                continue

            if current_label is not None and (line.startswith("http") or line.startswith("https")):
                videos.append({"url": line, "label": current_label})

    return videos

# =========================
# 7. 전체 파이프라인 (체크포인트 + JSONL append)
# =========================

def main(reset_url: str | None = None):
    if reset_url:
        reset_url_state(reset_url)

    video_list = parse_url_txt("url.txt")
    if not video_list:
        print("Error: No URLs found in url.txt or formatting is wrong.")
        return

    if TEST_ONLY_ONE_VIDEO:
        video_list = video_list[:1]
        print(f"Running TEST mode: Only processing {len(video_list)} video(s).")

    progress = load_progress()
    done = set(progress.get("done_urls", []))

    for video_info in tqdm(video_list, desc="Processing videos"):
        url = video_info["url"]
        label = video_info["label"]

        # 이미 끝낸 URL이면 스킵
        if url in done:
            print(f"\n[SKIP] Already processed: {url}")
            continue

        video_key = url_to_key(url)

        video_id = uuid.uuid4().hex
        video_path = f"{video_id}.mp4"

        print(f"\nDownloading: {url} (Label: {'fake' if label==0 else 'real'})")
        try:
            download_video(url, video_path)
        except Exception as e:
            print(f"Error downloading {url}: {e}")
            continue

        label_dir = FAKE_DIR if label == 0 else REAL_DIR

        print(f"Extracting {NUM_FRAMES_PER_VIDEO} frames...")
        frames = extract_frames(video_path, label_dir, video_key=video_key, num_frames=NUM_FRAMES_PER_VIDEO)

        if not frames:
            print(f"[WARN] No frames extracted for {url}")
            if os.path.exists(video_path):
                os.remove(video_path)
            continue

        # 15:5 split (20개 기준)
        for i, img_path in enumerate(frames):
            sample = build_sample(img_path, label, source_url=url, video_key=video_key)
            if i < 15:
                append_jsonl(TRAIN_JSONL, sample)
            else:
                append_jsonl(TEST_JSONL, sample)

        # 영상 파일 정리
        if os.path.exists(video_path):
            os.remove(video_path)

        # 여기까지 오면 "이 URL은 완료"로 체크포인트 저장
        done.add(url)
        progress["done_urls"] = sorted(done)
        save_progress(progress)
        print(f"[DONE] {url} saved to progress.")

    print("\nAll done.")
    print(f"- Train JSONL: {TRAIN_JSONL}")
    print(f"- Test  JSONL: {TEST_JSONL}")
    print(f"- Progress   : {PROGRESS_JSON}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset-url", type=str, default=None, help="특정 URL의 데이터/진행상태를 제거하고 재시작")
    args = parser.parse_args()
    main(reset_url=args.reset_url)
