import os
import json
import random
from google import genai

# Gemini Flash 모델 이름
MODEL_NAME = "gemini-2.5-flash"

# 환경변수에서 API 키 읽기
# 또는 직접 키 하드코딩(보안상 권장은 아님)
api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY 환경변수를 설정해 주세요.")

client = genai.Client(api_key=api_key)

# 프롬프트 템플릿
USER_PROMPTS = [
    "안녕?",
    "오늘 날씨 어때?",
    "점심 뭐 먹을까?",
    "영화 추천해줘",
    "주말에 뭐해?",
    "좋아하는 노래 있어?",
    "서울 가는 버스 시간 알아?",
    "운동 추천 좀 해줘",
    "피곤할 땐 뭐 하면 좋아?",
    "책 추천해줘"
]

def generate_dialogue_with_gemini(user_text: str) -> str:
    """Gemini에 user_text 묻고 응답 받기"""
    response = client.models.generate_content(
        model=MODEL_NAME,
        contents=user_text,
        # temperature나 max_output_tokens 옵션 넣을 수 있음
        config=genai.types.GenerateContentConfig(
            temperature=0.7,
            max_output_tokens=128
        )
    )
    return response.text

def generate_conversation(conv_id: int, max_turns: int = 2):
    """하나의 대화를 생성: user → assistant 형식, Gemini 사용"""
    dialogue = []
    for t in range(max_turns):
        user = random.choice(USER_PROMPTS)
        assistant = generate_dialogue_with_gemini(user)
        dialogue.append({"role": "user", "text": user})
        dialogue.append({"role": "assistant", "text": assistant})
    return {"id": f"conv{conv_id}", "dialogue": dialogue}

def generate_dataset_gemini(batch_size: int = 1000, output_dir: str = "gemini_datasets", start_index: int = 0):
    os.makedirs(output_dir, exist_ok=True)
    convos = []
    for i in range(start_index, start_index + batch_size):
        conv = generate_conversation(i, max_turns=2)
        convos.append(conv)
        if (i - start_index + 1) % 100 == 0:
            print(f"Generated {i - start_index + 1}/{batch_size} conversations")
    filename = os.path.join(output_dir, f"korean_gemini_chat_{start_index}_{start_index+batch_size-1}.json")
    with open(filename, "w", encoding="utf-8") as f:
        json.dump(convos, f, ensure_ascii=False, indent=2)
    print(f"✅ Saved dataset: {filename}")
    return filename

if __name__ == "__main__":
    # 예: 한번에 1000개 생성
    generate_dataset_gemini(batch_size=1000, start_index=0)