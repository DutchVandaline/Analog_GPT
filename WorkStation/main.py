import os
os.environ["HF_DATASETS_OFFLINE"] = "1"

from itertools import islice
from datasets import load_from_disk
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sentencepiece as spm
from tqdm.auto import tqdm

from Models.LatentMoE import LatentMoE
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
    NUM_EPOCHS = 10
    LR = 1e-4
    ACCUM_STEPS = 8

    MAX_SEQ_LEN = 256
    NUM_HEADS = 4
    EMBED_DIM = 192
    LATENT_DIM = 64
    MLP_DIM = 512
    NUM_LAYERS = 3
    DROPOUT = 0.1
    NUM_EXPERTS = 4
    EXPERTS_PER_TOKEN = 1
    BALANCE_LOSS_WEIGHT = 0.01

    # ---------- 토크나이저 ----------
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_bc.model")
    VOCAB_SIZE = tokenizer.GetPieceSize()

    # book_train_map = load_from_disk(r"C:\junha\Datasets\BookCorpus\train")
    # book_val_map = load_from_disk(r"C:\junha\Datasets\BookCorpus\val")
    wiki_train_map = load_from_disk(r"C:\junha\Datasets\WikiText103\train")
    wiki_val_map = load_from_disk(r"C:\junha\Datasets\WikiText103\val")
    train_iterable = wiki_train_map.to_iterable_dataset()
    val_iterable = wiki_val_map.to_iterable_dataset()


    train_dataset = StreamingDataset(train_iterable, tokenizer, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)
    val_dataset = StreamingDataset(val_iterable, tokenizer, max_seq_len=MAX_SEQ_LEN, stride=STRIDE)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
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

    # ---------- 모델 ----------
    model = LatentMoE(
        vocab_size=VOCAB_SIZE,
        max_seq_len=MAX_SEQ_LEN,
        embed_dim=EMBED_DIM,
        latent_dim=LATENT_DIM,
        mlp_dim=MLP_DIM,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
        num_heads=NUM_HEADS,
        num_experts=NUM_EXPERTS,
        experts_per_token=EXPERTS_PER_TOKEN,
        balance_loss_weight=BALANCE_LOSS_WEIGHT,
    ).to(device)
    model.lm_head.weight = model.token_embedding.weight

    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            betas=(0.9, 0.95), weight_decay=0.1)
    loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="mean")

    ckpt_dir = r"C:\junha\Git\BFG_2B\Checkpoints\BFG120M"
    os.makedirs(ckpt_dir, exist_ok=True)

    epoch_iter = tqdm(range(1, NUM_EPOCHS + 1), desc="Epochs")

    for epoch in epoch_iter:
        train_ppl, train_acc = train_step(model, train_dataloader,
                                          loss_fn, optimizer, device,
                                          accumulation_steps=ACCUM_STEPS,
                                          use_amp=True)

        val_ppl, val_acc, val_f1 = test_step(model, val_dataloader,
                                             loss_fn, device, use_amp=True)

        epoch_iter.set_postfix({
            "Train PPL": f"{train_ppl:.1f}",
            "Val PPL": f"{val_ppl:.1f}",
            "Val Acc": f"{val_acc * 100:.2f}%",
            "Val F1": f"{val_f1:.4f}"
        })

        torch.cuda.empty_cache()
        torch.save(model.state_dict(), os.path.join(ckpt_dir, f"120M_model_epoch_{epoch}.pt"))


if __name__ == "__main__":
    main()