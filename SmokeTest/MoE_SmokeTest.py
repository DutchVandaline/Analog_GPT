import torch
import sentencepiece as spm
from Models.LatentMoE import LatentMoE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_bc.model")

    VOCAB_SIZE = tokenizer.GetPieceSize()
    MAX_SEQ_LEN = 256
    NUM_HEADS = 8
    EMBED_DIM = 512
    LATENT_DIM = 128
    MLP_DIM = 1024
    NUM_LAYERS = 6
    DROPOUT = 0.1
    NUM_EXPERTS = 5
    EXPERTS_PER_TOKEN = 1
    BALANCE_LOSS_WEIGHT = 0.01


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
        balance_loss_weight=BALANCE_LOSS_WEIGHT
    ).to(device)

    model.eval()
    batch_size = 1
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, MAX_SEQ_LEN), device=device)
    with torch.no_grad():
        logits, balance_loss = model(input_ids)
    assert logits.shape == (batch_size, MAX_SEQ_LEN, VOCAB_SIZE)
    print("Forward OK · logits", logits.shape, f"balance_loss {balance_loss.item():.4f}")

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    targets = input_ids.clone()
    logits, balance_loss = model(input_ids)
    lm_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, VOCAB_SIZE), targets.view(-1)
    )
    total_loss = lm_loss + BALANCE_LOSS_WEIGHT * balance_loss
    total_loss.backward()
    optimizer.step()

    print(f"Backward OK · lm_loss {lm_loss.item():.4f} · balance_loss {balance_loss.item():.4f} · total_loss {total_loss.item():.4f}")
    print(f"Param count {param_count(model)/1e6:.1f}M")

if __name__ == "__main__":
    main()