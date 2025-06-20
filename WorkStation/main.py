import os

from Models.AnalogGPT import AnalogGPT

# HF Datasets offline 모드
os.environ["HF_DATASETS_OFFLINE"] = "1"

from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sentencepiece as spm
from tqdm.auto import tqdm

from Train_Step import train_step
from Test_Step import test_step
from WorkStation.StreamingDataset import StreamingDataset


def main():
    # ---------- 환경 ----------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    # ---------- 하이퍼파라미터 ----------
    BATCH_SIZE = 256
    STRIDE = 256
    NUM_WORKERS = 0
    NUM_EPOCHS = 30
    LR = 1e-3
    ACCUM_STEPS = 8

    MAX_SEQ_LEN = 16
    NUM_HEADS = 3
    EMBED_DIM = 12
    MLP_DIM = 48
    NUM_LAYERS = 4
    DROPOUT = 0.1

    # ---------- 토크나이저 로드 ----------
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\Analog_GPT\Tokenizers\spm_wiki2.model")
    VOCAB_SIZE = tokenizer.GetPieceSize()

    # ---------- 스트리밍 데이터 로드 및 셔플 ----------
    wiki_train_map = load_from_disk(r"C:\junha\Datasets\WikiText2\train")
    wiki_val_map   = load_from_disk(r"C:\junha\Datasets\WikiText2\val")

    train_iterable = (
        wiki_train_map
        .to_iterable_dataset()
        .shuffle(buffer_size=10_000, seed=42)
    )
    val_iterable = (
        wiki_val_map
        .to_iterable_dataset()
    )

    # ---------- 데이터셋 및 로더 ----------
    train_dataset = StreamingDataset(
        train_iterable, tokenizer,
        max_seq_len=MAX_SEQ_LEN, stride=STRIDE
    )
    val_dataset = StreamingDataset(
        val_iterable, tokenizer,
        max_seq_len=MAX_SEQ_LEN, stride=STRIDE
    )

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # 이미 iterable에서 섞음
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_dataloader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # ---------- 모델 초기화 ----------
    model = AnalogGPT(
        max_seq_len=MAX_SEQ_LEN,
        num_heads=NUM_HEADS,
        embedding_dim=EMBED_DIM,
        mlp_dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        mlp_size=MLP_DIM,
        vocab_size=VOCAB_SIZE
    ).to(device)

    # 옵티마이저 및 손실 함수
    optimizer = optim.AdamW(
        model.parameters(), lr=LR,
        betas=(0.9, 0.95), weight_decay=0.1
    )
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")

    # 체크포인트 디렉토리
    ckpt_dir = r"C:\junha\Git\Analog_GPT\Checkpoints\AnalogGPT_3k"
    os.makedirs(ckpt_dir, exist_ok=True)

    epoch_iter = tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs")

    for epoch in epoch_iter:
        train_ppl, train_acc = train_step(
            model, train_dataloader, loss_fn, optimizer, device,
            accumulation_steps=ACCUM_STEPS, use_amp=True
        )

        val_ppl, val_acc, val_f1 = test_step(
            model, val_dataloader, loss_fn, device, use_amp=True
        )

        epoch_iter.set_postfix({
            "Train PPL": f"{train_ppl:.1f}",
            "Val PPL":   f"{val_ppl:.1f}",
            "Val Acc":   f"{val_acc * 100:.2f}%",
            "Val F1":    f"{val_f1:.4f}"
        })

        torch.cuda.empty_cache()
        torch.save(
            model.state_dict(),
            os.path.join(ckpt_dir, f"3K_model_epoch_{epoch}.pt")
        )

if __name__ == "__main__":
    main()
