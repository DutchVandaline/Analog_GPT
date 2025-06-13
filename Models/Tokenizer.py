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
            text = row["content"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")

# RefinedWeb 샘플 생성
create_sample_file(
    "tiiuae/falcon-refinedweb",
    "train",
    output_file="refined_sample.txt",
    max_samples=1_000_000,
)

# SentencePiece 모델 학습 (vocab_size=35000)
spm.SentencePieceTrainer.Train(
    input=["refined_sample.txt"],
    model_prefix="spm_refined",
    vocab_size=35000,
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
    os.remove("refined_sample.txt")
except FileNotFoundError:
    pass

print("SentencePiece 모델 생성 완료 → spm_refined.model / spm_refined.vocab")
