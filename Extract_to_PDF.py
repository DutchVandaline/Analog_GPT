import os, json, shutil, torch, numpy as np, sentencepiece as spm
from Models.AnalogGPT import AnalogGPT

CHECKPOINT = r"C:\junha\Git\Analog_GPT\Checkpoints\AnalogGPT_3k\3K_model_epoch_30.pt"
TOKENIZER_MODEL = r"C:\junha\Git\Analog_GPT\Tokenizers\spm_wiki2.model"
SAVE = r"C:\junha\Git\Analog_GPT\Extracted"

DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN, NUM_HEADS, EMBED_DIM, MLP_DIM, NUM_LAYERS = 16, 1, 8, 32, 1

os.makedirs(os.path.join(SAVE, "weights"),      exist_ok=True)
os.makedirs(os.path.join(SAVE, "attn_scores"),  exist_ok=True)
os.makedirs(os.path.join(SAVE, "tokenizer"),    exist_ok=True)

state = torch.load(CHECKPOINT, map_location=DEV, weights_only=True)
vocab_size = state["token_embedding.weight"].shape[0]

model = AnalogGPT(
    vocab_size=vocab_size,
    max_seq_len=MAX_SEQ_LEN,
    embedding_dim=EMBED_DIM,
    num_layers=NUM_LAYERS,
    num_heads=NUM_HEADS,
    mlp_size=MLP_DIM,
    mlp_dropout=0.1,
    attn_dropout=0.1,
).to(DEV)
model.load_state_dict(state); model.eval()

for n, p in model.named_parameters():
    np.save(os.path.join(SAVE, "weights", n.replace(".", "_") + ".npy"),
            p.detach().cpu().numpy())

dummy = torch.randint(0, vocab_size, (1, MAX_SEQ_LEN), device=DEV)
x = model.token_embedding(dummy) + model.positional_embedding[:, :MAX_SEQ_LEN]

attn_scores = []
for blk in model.decoder.decoder_layers:
    y = blk.masked_msa_block.layer_norm(x)
    mask = torch.tril(torch.ones(y.size(1), y.size(1), device=DEV))
    out, w = blk.masked_msa_block.multihead_attn(
        y, y, y, attn_mask=mask,
        need_weights=True, average_attn_weights=False
    )
    attn_scores.append(w.detach().cpu().numpy())
    x = blk(x)

for li, sc in enumerate(attn_scores):
    b, h, _, _ = sc.shape
    for bi in range(b):
        for hi in range(h):
            np.save(os.path.join(SAVE, "attn_scores",
                   f"head_{li}_{bi}_{hi}.npy"), sc[bi, hi])

tok = spm.SentencePieceProcessor(); tok.Load(TOKENIZER_MODEL)
vocab = {i: tok.IdToPiece(i) for i in range(tok.GetPieceSize())}
with open(os.path.join(SAVE, "tokenizer", "vocab.json"),
          "w", encoding="utf-8") as f:
    json.dump(vocab, f, ensure_ascii=False, indent=2)

shutil.copy(TOKENIZER_MODEL,
            os.path.join(SAVE, "tokenizer",
                         os.path.basename(TOKENIZER_MODEL)))
