"""
TreeSelfAttentionLM and TreeSelfAttentionLM_GAT with Korean training loop example.
Now extended: dataset preparation using Stanza for Korean dependency parsing.
"""
import math
import argparse
import os
import random
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tree_llm import TreeSelfAttentionLM_GAT, TreeBiasConfig
import stanza

# -------------------------------
# Training function
# -------------------------------

def train_language_model(model: nn.Module, dataset: Dataset, batch_size: int = 8000,
                         epochs: int = 5, lr: float = 3e-4, device: str = "cpu"):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for step, (tokens, parents, mask) in enumerate(dataloader):
            tokens, parents, mask = tokens.to(device), parents.to(device), mask.to(device)
            logits, loss = model(tokens, parents, mask, targets=tokens)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            print(f"Epoch {epoch+1} Step {step}: loss {loss.item():.4f}")
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1} completed. Avg loss = {avg_loss:.4f}")

    return model

def sample_next_token(logits, temperature=1.0, top_k=0, top_p=1.0, past_tokens=None, repetition_penalty=1.2):
    logits = logits.clone()

    # 반복된 토큰 확률 낮추기
    if past_tokens is not None:
        for token_id in set(past_tokens.tolist()):
            logits[token_id] /= repetition_penalty

    # temperature 적용
    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    # top-k 필터링
    if top_k > 0:
        values, _ = torch.topk(probs, top_k)
        min_val = values[-1]
        probs[probs < min_val] = 0

    # top-p (nucleus) 필터링
    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        cutoff = cumulative_probs > top_p
        sorted_probs[cutoff] = 0
        probs = torch.zeros_like(probs).scatter(0, sorted_idx, sorted_probs)

    if probs.sum() == 0:
        probs = torch.softmax(logits, dim=-1)

    next_token_id = torch.multinomial(probs, num_samples=1).item()
    return next_token_id




def stream_generate_text(model, dataset, prompt, max_new_tokens=50, device="cpu"):
    model.eval()
    id_to_word = {v: k for k, v in dataset.vocab.items()}

    doc = dataset.nlp(prompt)
    tokens = [word.text for sentence in doc.sentences for word in sentence.words]
    token_ids = [dataset.vocab.get(tok, 0) for tok in tokens]

    parents = [i - 1 if i > 0 else -1 for i in range(len(token_ids))]

    past_tokens = torch.tensor([token_ids], device=device)
    past_parents = torch.tensor([parents], device=device)
    past_mask = past_tokens != 0

    print("▶ Generated:", prompt, end=" ", flush=True)

    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(past_tokens, past_parents, past_mask)

        next_token_logits = logits[0, past_tokens.size(1) - 1]
        next_token_id = sample_next_token(
            next_token_logits,
            temperature=0.3,                  # 다양성 조절 (1.0 기본, 0.7~1.0 추천)
            top_k=5,                          # 상위 k개 후보만 고려
            top_p=0.95,                       # nucleus sampling 비율
            past_tokens=past_tokens[0],       # 지금까지 생성된 문맥
            repetition_penalty=1.2            # 1.0이면 무효, 1.2~1.5 추천
        )
        if next_token_id == 0:
            break

        past_tokens = torch.cat([past_tokens, torch.tensor([[next_token_id]], device=device)], dim=1)
        past_parents = torch.cat([past_parents, torch.tensor([[past_tokens.size(1)-2]], device=device)], dim=1)
        past_mask = past_tokens != 0

        word = id_to_word.get(next_token_id, "<unk>")
        print(word, end=" ", flush=True)

    print("\n✅ Done")


def generate_text_fast(model, dataset, prompt, max_new_tokens=50, device="cpu"):
    model.eval()
    id_to_word = {v: k for k, v in dataset.vocab.items()}

    # 1. 프롬프트 → 토큰 인코딩
    doc = dataset.nlp(prompt)
    tokens = [word.text for sentence in doc.sentences for word in sentence.words]
    token_ids = [dataset.vocab.get(tok, 0) for tok in tokens]

    # parents 배열 (간단히 선형 체인으로 처리)
    parents = [i - 1 if i > 0 else -1 for i in range(len(token_ids))]

    input_tensor = torch.tensor([token_ids], device=device)
    parents_tensor = torch.tensor([parents], device=device)
    mask_tensor = (input_tensor != 0).to(device)

    # 2. hidden state 캐시 초기화
    past_tokens = input_tensor
    past_parents = parents_tensor
    past_mask = mask_tensor

    # 3. 반복적으로 새 토큰 생성
    for _ in range(max_new_tokens):
        with torch.no_grad():
            logits, _ = model(past_tokens, past_parents, past_mask)

        # 마지막 토큰의 예측 확률만 사용
        next_token_logits = logits[0, len(past_tokens[0]) - 1]
        next_token_id = next_token_logits.argmax().item()

        if next_token_id == 0:  # PAD 예측 시 종료
            break

        # 새 토큰 추가
        past_tokens = torch.cat([past_tokens, torch.tensor([[next_token_id]], device=device)], dim=1)
        past_parents = torch.cat([past_parents, torch.tensor([[len(past_tokens[0]) - 2]], device=device)], dim=1)
        past_mask = past_tokens != 0

    # 4. id → 단어 변환
    generated = [id_to_word.get(idx, "<unk>") for idx in past_tokens[0].tolist()]
    return " ".join(generated)



# ==========================
# Dataset Class
# ==========================

class KoreanTreeDataset(Dataset):
    def __init__(self, corpus=None, vocab=None, max_len=None,
                 tokens=None, parents=None, mask=None):
        """
        corpus: raw sentences
        vocab: word -> id dict
        max_len: max sequence length
        tokens, parents, mask: precomputed tensors (for reload)
        """
        if tokens is not None and parents is not None and mask is not None:
            # 이미 저장된 dataset 불러오기
            self.tokens = tokens
            self.parents = parents
            self.mask = mask
            self.vocab = vocab
            self.max_len = max_len
            return

        # Stanza tokenizer
        self.nlp = stanza.Pipeline(lang="ko",
                                   processors="tokenize,pos,lemma,depparse",
                                   tokenize_pretokenized=False)

        tokens_list, parents_list = [], []
        for sent in corpus:
            doc = self.nlp(sent)
            for sentence in doc.sentences:
                toks = [word.text for word in sentence.words]
                prnts = [w.head - 1 if w.head > 0 else -1 for w in sentence.words]
                tokens_list.append(toks)
                parents_list.append(prnts)

        # vocab
        if vocab is None:
            vocab = {}
        self.vocab = vocab

        def get_id(word):
            if word not in self.vocab:
                self.vocab[word] = len(self.vocab) + 1  # 0 reserved for padding
            return self.vocab[word]

        encoded = [[get_id(tok) for tok in toks] for toks in tokens_list]

        if max_len is None:
            max_len = max(len(seq) for seq in encoded)
        self.max_len = max_len

        def pad(seq, pad_value):
            return seq + [pad_value] * (max_len - len(seq))

        self.tokens = torch.tensor([pad(seq, 0) for seq in encoded])
        self.parents = torch.tensor([pad(p, -1) for p in parents_list])
        self.mask = self.tokens != 0

    def __len__(self):
        return self.tokens.size(0)

    def __getitem__(self, idx):
        return self.tokens[idx], self.parents[idx], self.mask[idx]


# ==========================
# Main Runner
# ==========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TreeSelfAttentionLM_Korean Runner")
    parser.add_argument("--mode", type=str, default="predict",
                        choices=["train", "predict"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--prompt", type=str, default="문재인 대통령은")
    parser.add_argument("--max_new_tokens", type=int, default=50)
    parser.add_argument("--continue_train", action="store_true")
    parser.add_argument("--use_excel", action="store_true")
    parser.add_argument("--excel_path", type=str, default="./datasets/dia.xlsx")
    parser.add_argument("--checkpoint", type=str,
                        default="./train_models/tree_llm/checkpoint.pth",
                        help="모델+데이터셋 체크포인트 경로")
    args = parser.parse_args()

    torch.manual_seed(42)

    # -------------------------------
    # 데이터셋 준비
    # -------------------------------
    if os.path.exists(args.checkpoint) and args.continue_train:
        print(f"Loading dataset+vocab from {args.checkpoint}")
        ckpt = torch.load(args.checkpoint, map_location="cpu")
        dataset = KoreanTreeDataset(
            vocab=ckpt["vocab"],
            max_len=ckpt["max_len"],
            tokens=ckpt["tokens"],
            parents=ckpt["parents"],
            mask=ckpt["mask"]
        )
        vocab = ckpt["vocab"]
    else:
        corpus = []
        if args.use_excel:
            import pandas as pd
            df = pd.read_excel(args.excel_path)
            all_sentences, chat_flow = [], ""
            for idx, row in df.iterrows():
                for role in ["사람문장1", "시스템문장1",
                             "사람문장2", "시스템문장2",
                             "사람문장3", "시스템문장3"]:
                    if role in row and isinstance(row[role], str) and row[role].strip():
                        sentence = row[role].strip()
                        chat_flow += f"=> {sentence} "
                all_sentences.append(chat_flow)
                chat_flow = ""
            if len(all_sentences) > 1000:
                start_idx = random.randint(0, max(0, len(all_sentences) - 1000))
                corpus = all_sentences[start_idx:start_idx + 1000]
            else:
                corpus = all_sentences
        else:
            with open("./datasets/korean_sentences.txt", "r", encoding="utf-8") as f:
                for i in range(random.randint(0, 100000)):
                    f.readline()
                for i in range(1000):
                    line = f.readline().strip()
                    corpus.append(line)

        dataset = KoreanTreeDataset(corpus, max_len=128)
        vocab = dataset.vocab

    print(f"Dataset loaded. Vocab size = {len(vocab)}, Max_len = {dataset.max_len}")

    V = 120000
    L = dataset.max_len
    print(L)

    model = TreeSelfAttentionLM_GAT(
        vocab_size=V, d_model=256, n_heads=4,
        n_layers=8, dropout=0.1, max_len=128, tree_gnn_layers=4
    )

    # -------------------------------
    # Training
    # -------------------------------
    if args.mode == "train":
        if args.continue_train and os.path.exists(args.checkpoint):
            print(f"Continuing training from {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
        else:
            print("Starting new training")

        trained_model = train_language_model(
            model, dataset,
            batch_size=args.batch_size,
            epochs=args.epochs,
            lr=args.lr,
            device="cpu"
        )
        os.makedirs(os.path.dirname(args.checkpoint), exist_ok=True)
        torch.save({
            "model_state": trained_model.state_dict(),
            "vocab": dataset.vocab,
            "max_len": dataset.max_len,
            "tokens": dataset.tokens,
            "parents": dataset.parents,
            "mask": dataset.mask
        }, args.checkpoint)
        print(f"Checkpoint saved to {args.checkpoint}")

    # -------------------------------
    # Prediction
    # -------------------------------
    elif args.mode == "predict":
        if os.path.exists(args.checkpoint):
            print(f"Loading model from {args.checkpoint}")
            ckpt = torch.load(args.checkpoint, map_location="cpu")
            model.load_state_dict(ckpt["model_state"])
            dataset = KoreanTreeDataset(
                vocab=ckpt["vocab"],
                max_len=ckpt["max_len"],
                tokens=ckpt["tokens"],
                parents=ckpt["parents"],
                mask=ckpt["mask"]
            )
        else:
            print("⚠️ Warning: 체크포인트 없음. 랜덤 초기화 모델 사용")

        model.eval()
        with torch.no_grad():
            while True:
                text = input("Enter a prompt (or 'exit' to quit): ")
                if text.lower() == "exit":
                    break
                print("===================================")
                print(f"Prompt: {text}")
                print("Generated text:")
                stream_generate_text(model, dataset, "=> " + text,
                                     max_new_tokens=args.max_new_tokens, device="cpu")
                print("===================================")
