import os
import json
import urllib.request
import xml.etree.ElementTree as ET
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Hannanum


def make_url(word: str) -> str:
    url = "https://krdict.korean.go.kr/api/search?key=B241048B698F8FDDC1B3033C3B061B66&type_search=search&part=word&q="
    return url + urllib.parse.quote(word)


def get_data(url: str) -> str:
    response = urllib.request.urlopen(url)
    return response.read().decode("utf-8")


def parse_data(data: str) -> list:
    root = ET.fromstring(data)
    items = root.findall("item")
    result = []
    for item in items:
        word = item.find("word").text
        definition = item.find("sense").find("definition").text
        result.append((word, definition))
    return result


def get_word_definition(word: str) -> list:
    url = make_url(word)
    data = get_data(url)
    return parse_data(data)


if __name__ == "__main__":
    hannanum = Hannanum()

    # 🔹 최신 모델 & 단어 세트 불러오기
    latest_model = None
    latest_iter = 0
    words = set()

    # 모델 파일 탐색 (word2vec_숫자.model)
    model_files = [f for f in os.listdir(".") if f.startswith("word2vec_") and f.endswith(".model")]
    if model_files:
        # 가장 최신 파일 찾기
        latest_model = max(model_files, key=lambda x: int(x.split("_")[1].split(".")[0]))
        latest_iter = int(latest_model.split("_")[1].split(".")[0])

        print(f"✅ 최신 모델 불러오기: {latest_model}")
        model = Word2Vec.load(latest_model)

        # 단어 세트 불러오기
        if os.path.exists("word_set.json"):
            with open("word_set.json", "r", encoding="utf-8") as f:
                words = set(json.load(f))
        else:
            words = set()

    else:
        print("⚠️ 기존 모델 없음 → 새로 생성")
        model = Word2Vec(vector_size=0xff, window=0xf, min_count=1, workers=12, sg=0)
        model.build_vocab([["시작"]])  # seed
        model.train([["시작"]], total_examples=1, epochs=1)
        words = set()

    # 시작 단어
    start_word = "사과"
    items = [start_word] if start_word not in words else list(words)
    words.add(start_word)

    max_iterations = 10000000000000000
    iteration = latest_iter

    while len(items) > 0:
        if iteration >= max_iterations:
            break

        iteration += 1
        item = items.pop(0)
        print(f"item : {item}")
        data = get_word_definition(item)

        if len(data) == 0:
            continue

        for word, definition in data:
            word = "".join(filter(lambda x: ord("가") <= ord(x) <= ord("힣"), word))
            definition = "".join(
                filter(lambda x: ord("가") <= ord(x) <= ord("힣") or x == " ", definition)
            )

            extras = hannanum.morphs(definition)
            for extra in extras:
                if extra not in words:
                    words.add(extra)
                    items.append(extra)

            if word not in words:
                words.add(word)
                items.append(word)

            print(f"items count : {len(items)}")
            tokens = hannanum.morphs(definition)
            model.build_vocab([tokens], update=True)
            model.train([tokens], total_examples=1, epochs=1)

        # 🔹 10번마다 모델 + 단어 세트 저장
        if iteration % 10 == 0:
            print(f"iteration : {iteration}, words : {len(words)}, items : {len(items)}")
            model.save(f"word2vec_{iteration}.model")
            with open("word_set.json", "w", encoding="utf-8") as f:
                json.dump(list(words), f, ensure_ascii=False, indent=2)

        print("====================================")

    # 예시: 결과 확인
    print(model.wv.most_similar("꽃"))
    print(model.wv.most_similar("날"))
    print(model.wv.most_similar("아침에"))
    print(model.wv.most_similar("뜨자"))
