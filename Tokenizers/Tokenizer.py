import sentencepiece as spm
import os
from datasets import disable_caching, load_dataset

disable_caching()

def create_sample_file(dataset_name, split, config_name=None, output_file=None, max_samples=1_000_000, cache_dir="./datasets"):
    stream = load_dataset(
        dataset_name,
        config_name,
        split=split,
        streaming=True,
        cache_dir=cache_dir,
        trust_remote_code=True,
    )
    stream = stream.shuffle(buffer_size=10_000, seed=42)

    with open(output_file, "w", encoding="utf-8") as f:
        for i, row in zip(range(max_samples), stream):
            # WikiText-2는 'text' 필드에 원문이 들어 있습니다
            text = row["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")

# WikiText-2 샘플 생성
create_sample_file(
    dataset_name="wikitext",
    config_name="wikitext-2-raw-v1",
    split="train",
    output_file="wikitext2_sample.txt",
    max_samples=1_000_000,   # 필요에 따라 줄여도 좋습니다
)

# SentencePiece 모델 학습 (vocab_size=150)
spm.SentencePieceTrainer.Train(
    input=["wikitext2_sample.txt"],
    model_prefix="spm_wiki2",
    vocab_size=300,
    character_coverage=0.9995,
    model_type="bpe",
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3,
    normalization_rule_name="nfkc",
)

# 임시 파일 삭제
try:
    os.remove("wikitext2_sample.txt")
except FileNotFoundError:
    pass

print("SentencePiece 모델 생성 완료 → spm_wiki2.model / spm_wiki2.vocab")
