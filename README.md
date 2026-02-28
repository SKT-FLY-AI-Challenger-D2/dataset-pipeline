# Youtube-Fake-Clue Dataset Pipeline

Youtube Deepfake/AI manipulated Image Dataset Labeling Pipeline

GPT,Gemini,Claude Cooperated 자동 레이블링을 통해  
설명(reasoning)을 생성하여  
FakeVLM 학습 포맷으로 train.json / test.json을 생성한다.

---

## 🔥 목적

- 한국어 기반 Deepfake 설명 데이터셋 구축
- FakeClue 방식 차용 (설명 중심 레이블링)
- GPT-5-mini, Gemini-3.0-Flash, Claude-Haiku-4.5 기반 자동 레이블링
- FakeVLM 학습 포맷과 호환

---

## 📂 디렉토리 구조

```
playground
└── data
    ├── fake
    ├── real
    ├── train.json
    └── test.json
```



- fake / real : 프레임 이미지 저장
- train.json / test.json : 학습용 메타데이터

---

## 📦 설치

```bash
pip install yt-dlp opencv-python pillow tqdm openai
```
```
JSON 생성 포맷
{
  "image": "fake/xxx.png",
  "label": 0,
  "cate": "deepfake",
  "width": 256,
  "height": 256,
  "conversations": [
    {
      "from": "human",
      "value": "<image>is this image fake or real?"
    },
    {
      "from": "gpt",
      "value": "this image is fake, texture: ..."
    }
  ]
}
```

label: 0 = fake, 1 = real
cate: deepfake / real

