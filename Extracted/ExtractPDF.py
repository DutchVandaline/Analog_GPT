import os, json, numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, PageBreak
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

BASE_DIR    = r"C:\junha\Git\Analog_GPT\Extracted"
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
ATTN_DIR    = os.path.join(BASE_DIR, "attn_scores")
TOKEN_DIR   = os.path.join(BASE_DIR, "tokenizer")
OUTPUT_PDF  = os.path.join(BASE_DIR, "extracted_all.pdf")

doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4)
styles, elements = getSampleStyleSheet(), []

# ---------- 1. Weights ----------
elements.append(Paragraph("1. Weights", styles["Heading2"]))
for fname in sorted(os.listdir(WEIGHTS_DIR)):
    if not fname.endswith(".npy"):
        continue
    arr = np.load(os.path.join(WEIGHTS_DIR, fname))
    elements.append(Paragraph(fname, styles["Heading4"]))

    if arr.ndim == 1:          # 1D
        slices = [arr]
        slice_titles = [None]

    elif arr.ndim == 2:        # 2D
        slices = [arr]
        slice_titles = [None]

    elif arr.ndim == 3:        # 3D → 모든 첫축 슬라이스
        slices = [arr[i] for i in range(arr.shape[0])]
        slice_titles = [f"slice {i}" for i in range(arr.shape[0])]

    else:                      # 4D 이상은 그대로 flatten
        flat = arr.reshape(arr.shape[0], -1)
        slices = [flat]
        slice_titles = [f"flattened (ndim={arr.ndim})"]

    for slc, title in zip(slices, slice_titles):
        if title:
            elements.append(Paragraph(title, styles["Normal"]))

        data = [[f"{v:.4f}" for v in row] for row in (slc if slc.ndim==2 else slc[np.newaxis, :])]
        tbl = Table(data)
        tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        elements.extend([tbl, Spacer(1,12)])
elements.append(PageBreak())

# ---------- 2. Attention Scores ----------
elements.append(Paragraph("2. Attention Scores", styles["Heading2"]))

for fname in sorted(os.listdir(ATTN_DIR)):
    if not fname.endswith(".npy"):
        continue
    attn = np.load(os.path.join(ATTN_DIR, fname))   # ndim 2‥4 모두 가능

    # 2D  : (seq, seq)
    if attn.ndim == 2:
        batches, heads = 1, 1
        matrices = [(0, 0, attn)]

    # 3D  : (heads, seq, seq)
    elif attn.ndim == 3:
        batches, heads = 1, attn.shape[0]
        matrices = [(0, h, attn[h]) for h in range(heads)]

    # 4D  : (batch, heads, seq, seq)
    elif attn.ndim == 4:
        batches, heads = attn.shape[:2]
        matrices = [
            (b, h, attn[b, h])            # (seq, seq)
            for b in range(batches)
            for h in range(heads)
        ]

    else:                                  # 이상 차원은 건너뜀
        elements.append(Paragraph(f"{fname}: skipping (ndim={attn.ndim})", styles["Normal"]))
        elements.append(Spacer(1, 12))
        continue

    for b, h, mat in matrices:
        label = f"{fname} — batch {b}, head {h}" if (batches > 1 or heads > 1) else fname
        elements.append(Paragraph(label, styles["Heading4"]))
        data = [[f"{v:.4f}" for v in row] for row in mat]   # mat 은 2D 확실
        tbl = Table(data)
        tbl.setStyle(TableStyle([
            ("GRID",  (0,0), (-1,-1), 0.3, colors.grey),
            ("FONTSIZE",(0,0),(-1,-1), 6),
            ("TOPPADDING",(0,0),(-1,-1),2),
            ("BOTTOMPADDING",(0,0),(-1,-1),2),
        ]))
        elements.extend([tbl, Spacer(1,12)])
elements.append(PageBreak())

# ---------- 3. Tokenizer Vocabulary ----------
elements.append(Paragraph("3. Tokenizer Vocabulary", styles["Heading2"]))
with open(os.path.join(TOKEN_DIR, "vocab.json"), encoding="utf-8") as f:
    vocab = json.load(f)
pairs = sorted(vocab.items(), key=lambda x: int(x[0]))
for i in range(0, len(pairs), 50):
    chunk = pairs[i:i+50]
    data = [[idx, tok] for idx, tok in chunk]
    tbl = Table(data, colWidths=[40, 200])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 6),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    elements.extend([tbl, Spacer(1,12)])

doc.build(elements)
print(f"PDF saved → {OUTPUT_PDF}")
