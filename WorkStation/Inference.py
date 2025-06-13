import torch
import torch.nn.functional as F
import sentencepiece as spm
from Models.AnalogGPT import AnalogGPT

def load_model(checkpoint_path: str, vocab_size: int, max_seq_len: int,
               num_heads: int, embed_dim: int, mlp_dim: int,
               num_layers: int, dropout: float, device: torch.device) -> AnalogGPT:
    model = AnalogGPT(
        max_seq_len=max_seq_len,
        num_heads=num_heads,
        embedding_dim=embed_dim,
        mlp_dropout=dropout,
        num_layers=num_layers,
        mlp_size=mlp_dim,
        vocab_size=vocab_size,
    ).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()
    model.max_seq_len = max_seq_len
    return model

@torch.no_grad()
def generate(model: AnalogGPT,
             tokenizer: spm.SentencePieceProcessor,
             prompt: str,
             max_generated: int = 32,
             temperature: float = 1.0,
             top_k: int = 0,
             top_p: float = 0.0,
             device: torch.device = torch.device("cpu")) -> str:
    tokens = tokenizer.encode(prompt, out_type=int)
    tokens = tokens[-model.max_seq_len:]
    pad_len = model.max_seq_len - len(tokens)
    input_ids = torch.tensor([tokens + [tokenizer.eos_id()] * pad_len], device=device)
    generated = tokens.copy()

    for _ in range(max_generated):
        logits = model(input_ids)
        logits = logits[0, -1] / (temperature if temperature > 0 else 1.0)
        if top_k > 0:
            values, _ = torch.topk(logits, top_k)
            min_val = values[-1]
            logits[logits < min_val] = -float('Inf')
        if top_p > 0.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            probs = F.softmax(sorted_logits, dim=-1)
            cum_probs = probs.cumsum(dim=-1)
            sorted_indices_to_remove = cum_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits[indices_to_remove] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        new_seq = generated[-model.max_seq_len:]
        input_ids = torch.tensor([new_seq], device=device)

    return tokenizer.decode(generated)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_CKPT = r"C:\junha\Git\Analog_GPT\Checkpoints\AnalogGPT_3k\3K_model_epoch_30.pt"
    SP_MODEL   = r"C:\junha\Git\Analog_GPT\Tokenizers\spm_wiki2.model"

    tokenizer = spm.SentencePieceProcessor()
    tokenizer.Load(SP_MODEL)
    VOCAB_SIZE = tokenizer.GetPieceSize()

    model = load_model(
        checkpoint_path=MODEL_CKPT,
        vocab_size=VOCAB_SIZE,
        max_seq_len=16,
        num_heads=1,
        embed_dim=8,
        mlp_dim=32,
        num_layers=1,
        dropout=0.1,
        device=device,
    )

    prompt = "The history of artificial intelligence"
    result = generate(
        model, tokenizer, prompt,
        max_generated=50,
        temperature=0.8,
        top_k=10,
        top_p=0.9,
        device=device
    )
    print(result)
