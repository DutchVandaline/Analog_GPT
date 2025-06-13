import os, shutil, torch, numpy as np, json, sentencepiece as spm
from Models.AnalogGPT import AnalogGPT

CHECKPOINT = r"C:\junha\Git\Analog_GPT\Checkpoints\AnalogGPT_3k\3K_model_epoch_30.pt"
TOKENIZER_MODEL = r"C:\junha\Git\Analog_GPT\Tokenizers\spm_wiki2.model"
SAVE = r"C:\junha\Git\Analog_GPT\Extracted"
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_SEQ_LEN, NUM_HEADS, EMBED_DIM, MLP_DIM, NUM_LAYERS = 16, 1, 8, 32, 1

os.makedirs(f"{SAVE}/weights", exist_ok=True)
os.makedirs(f"{SAVE}/attn_scores", exist_ok=True)
os.makedirs(f"{SAVE}/tokenizer", exist_ok=True)

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
    attn_dropout=0.1
).to(DEV)
model.load_state_dict(state)
model.eval()

for name, param in model.named_parameters():
    np.save(f"{SAVE}/weights/{name.replace('.', '_')}.npy",
            param.detach().cpu().numpy())

dummy = torch.randint(0, vocab_size, (1, MAX_SEQ_LEN), device=DEV)
pos_emb = model.positional_embedding[:, :dummy.size(1), :]
x = model.token_embedding(dummy) + pos_emb

attn_scores = []
for i, block in enumerate(model.decoder.decoder_layers):
    normed = block.masked_msa_block.layer_norm(x)
    mask = torch.tril(torch.ones(normed.size(1),
                                 normed.size(1),
                                 device=DEV))
    out_attn, weights = block.masked_msa_block.multihead_attn(
        query=normed,
        key=normed,
        value=normed,
        attn_mask=mask,
        need_weights=True,
        average_attn_weights=False
    )
    attn_scores.append(weights.detach().cpu().numpy())
    x_res1 = out_attn + x
    mlp_out = block.mlp_block(x_res1)
    x = mlp_out + x_res1

for idx, scores in enumerate(attn_scores):
    np.save(f"{SAVE}/attn_scores/head_{idx}.npy", scores)

tokenizer = spm.SentencePieceProcessor()
tokenizer.Load(TOKENIZER_MODEL)
mapping = {i: tokenizer.IdToPiece(i)
           for i in range(tokenizer.GetPieceSize())}
with open(f"{SAVE}/tokenizer/vocab.json", "w",
          encoding="utf-8") as f:
    json.dump(mapping, f, ensure_ascii=False, indent=2)

shutil.copy(TOKENIZER_MODEL,
            f"{SAVE}/tokenizer/{os.path.basename(TOKENIZER_MODEL)}")
