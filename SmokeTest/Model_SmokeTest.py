import torch
import sentencepiece as spm

from Models.AnalogGPT import AnalogGPT

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def param_count(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def main():
    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(r"C:\junha\Git\BFG_2B\Tokenizer\spm_bc.model")

    VOCAB_SIZE = 150
    MAX_SEQ_LEN = 16
    NUM_HEADS = 1
    EMBED_DIM = 8
    MLP_DIM = 32
    NUM_LAYERS = 1
    DROPOUT = 0.1


    model = AnalogGPT(
        max_seq_len=MAX_SEQ_LEN,
        num_heads=NUM_HEADS,
        embedding_dim=EMBED_DIM,
        mlp_dropout=DROPOUT,
        num_layers=NUM_LAYERS,
        mlp_size = MLP_DIM,
        vocab_size=VOCAB_SIZE
    ).to(device)

    model.eval()
    batch_size = 1
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, MAX_SEQ_LEN), device=device)
    with torch.no_grad():
        logits = model(input_ids)
    assert logits.shape == (batch_size, MAX_SEQ_LEN, VOCAB_SIZE)
    print("Forward OK · logits", logits.shape)

    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    targets = input_ids.clone()
    logits = model(input_ids)
    lm_loss = torch.nn.functional.cross_entropy(
        logits.view(-1, VOCAB_SIZE), targets.view(-1)
    )
    total_loss = lm_loss
    total_loss.backward()
    optimizer.step()

    print(f"Backward OK · lm_loss {lm_loss.item():.4f} · total_loss {total_loss.item():.4f}")
    print(f"Param count {param_count(model)}")

if __name__ == "__main__":
    main()