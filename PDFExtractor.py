import os, torch, numpy as np, sentencepiece as spm
from transformers import PreTrainedTokenizerFast
from fpdf import FPDF


def _ensure(p):
    os.makedirs(p, exist_ok=True)


def _save_txt(arr, path, per_line=12):
    flat = arr.flatten()
    with open(path, "w", encoding="utf-8") as f:
        for i in range(0, len(flat), per_line):
            f.write(":".join(f"{v:.6e}" for v in flat[i:i + per_line]) + "\n")


class _WeightPDF(FPDF):
    def __init__(self):
        super().__init__("P", "mm", "A4")
        self.set_font("Courier", size=7)

    def add_tensor(self, name, t, per_line=8):
        self.add_page()
        self.set_font_size(10)
        self.multi_cell(0, 4, name)
        self.set_font_size(7)
        flat = t.flatten().cpu().numpy()
        for i in range(0, len(flat), per_line):
            self.multi_cell(0, 3, ":".join(f"{v:.5e}" for v in flat[i:i + per_line]))


def _spm_to_vocab(model_path, vocab_txt):
    sp = spm.SentencePieceProcessor()
    sp.load(model_path)
    with open(vocab_txt, "w", encoding="utf-8") as f:
        for idx in range(sp.get_piece_size()):
            f.write(f"{idx}\t{sp.id_to_piece(idx)}\n")


def _emit_tables(src_txt, csv_out, md_out, cols=12):
    with open(src_txt, encoding="utf-8") as f:
        cells = [ln.strip().replace("\t", ":") for ln in f if ln.strip()]

    with open(csv_out, "w", encoding="utf-8") as g:
        for r in range(0, len(cells), cols):
            g.write(":".join(cells[r:r + cols]) + "\n")

    with open(md_out, "w", encoding="utf-8") as g:
        g.write("| " + " | ".join([f"col{i + 1}" for i in range(cols)]) + " |\n")
        g.write("|" + " --- |" * cols + "\n")
        for r in range(0, len(cells), cols):
            row = cells[r:r + cols] + [""] * (cols - len(cells[r:r + cols]))
            g.write("| " + " | ".join(row) + " |\n")


def export_model(
    checkpoint,
    tokenizer_path,
    output_dir="exported_model",
    make_pdf=True,
):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state_dict = state if isinstance(state, dict) else state.state_dict()

    tok_dir = os.path.join(output_dir, "tokenizer")
    _ensure(tok_dir)
    tokenizer = None

    if os.path.isdir(tokenizer_path):
        spm_models = [p for p in os.listdir(tokenizer_path) if p.endswith(".model")]
        if spm_models:
            _spm_to_vocab(os.path.join(tokenizer_path, spm_models[0]), os.path.join(tok_dir, "vocab.txt"))
        else:
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
    elif tokenizer_path.endswith(".model"):
        _spm_to_vocab(tokenizer_path, os.path.join(tok_dir, "vocab.txt"))
    else:
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

    if tokenizer:
        tokenizer.save_pretrained(tok_dir)
        with open(os.path.join(tok_dir, "vocab.txt"), "w", encoding="utf-8") as f:
            for idx in range(tokenizer.vocab_size):
                f.write(f"{idx}\t{tokenizer.convert_ids_to_tokens(idx)}\n")

    _emit_tables(
        os.path.join(tok_dir, "vocab.txt"),
        os.path.join(tok_dir, "vocab_6col.csv"),
        os.path.join(tok_dir, "vocab_6col.md"),
    )

    wtxt = os.path.join(output_dir, "weights_txt")
    _ensure(wtxt)
    pdf = _WeightPDF() if make_pdf else None
    for name, tensor in state_dict.items():
        _save_txt(tensor.cpu().numpy(), os.path.join(wtxt, f"{name}.txt"))
        if pdf:
            pdf.add_tensor(name, tensor)

    if pdf:
        pdf.output(os.path.join(output_dir, "weights.pdf"))

    print("Done →", os.path.abspath(output_dir))


export_model(
    checkpoint=r"C:\junha\Git\Analog_GPT\Checkpoints\AnalogGPT_15k\15K_model_epoch_100.pt",
    tokenizer_path=r"C:\junha\Git\Analog_GPT\Tokenizers\spm_wiki2.model",
    output_dir="analog_book",
    make_pdf=True,
)
