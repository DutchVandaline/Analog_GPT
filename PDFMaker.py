import os
import numpy as np
import json
from reportlab.lib.pagesizes import A4
from reportlab.platypus import (
    SimpleDocTemplate, Table, TableStyle,
    Paragraph, Spacer, PageBreak
)
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet

BASE_DIR    = r"C:\junha\Git\Analog_GPT\Extracted"
WEIGHTS_DIR = os.path.join(BASE_DIR, "weights")
ATTN_DIR    = os.path.join(BASE_DIR, "attn_scores")
TOKEN_DIR   = os.path.join(BASE_DIR, "tokenizer")
OUTPUT_PDF  = os.path.join(BASE_DIR, "extracted_all.pdf")

doc = SimpleDocTemplate(OUTPUT_PDF, pagesize=A4)
styles = getSampleStyleSheet()
elements = []

# 1. Weights
elements.append(Paragraph("1. Weights", styles["Heading2"]))
for fname in sorted(os.listdir(WEIGHTS_DIR)):
    if not fname.endswith(".npy"):
        continue
    arr = np.load(os.path.join(WEIGHTS_DIR, fname))
    elements.append(Paragraph(fname, styles["Heading4"]))

    # 1D 벡터
    if arr.ndim == 1:
        data = [[f"{val:.4f}" for val in arr]]

    # 2D 행렬
    elif arr.ndim == 2:
        data = [[f"{val:.4f}" for val in row] for row in arr]

    # 3D 텐서: 첫 슬라이스만 사용
    elif arr.ndim == 3:
        arr2d = arr[0]
        elements.append(Paragraph("(3D tensor, showing slice [0])", styles["Normal"]))
        data = [[f"{val:.4f}" for val in row] for row in arr2d]

    else:
        # 그 이상 차원은 스킵
        elements.append(Paragraph(f"Skipping array with ndim={arr.ndim}", styles["Normal"]))
        elements.append(Spacer(1, 12))
        continue

    tbl = Table(data)
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 6),
        ("TOPPADDING", (0,0), (-1,-1), 2),
        ("BOTTOMPADDING", (0,0), (-1,-1), 2),
    ]))
    elements.extend([tbl, Spacer(1,12)])
elements.append(PageBreak())

# 2. Attention Scores
elements.append(Paragraph("2. Attention Scores", styles["Heading2"]))
for fname in sorted(os.listdir(ATTN_DIR)):
    if not fname.endswith(".npy"):
        continue
    arr4d = np.load(os.path.join(ATTN_DIR, fname))  # (batch, heads, seq, seq)
    batch, heads, seq, _ = arr4d.shape

    for h in range(heads):
        elements.append(Paragraph(f"{fname} — head {h}", styles["Heading4"]))
        arr2d = arr4d[0, h]
        data = [[f"{val:.4f}" for val in row] for row in arr2d]
        tbl = Table(data)
        tbl.setStyle(TableStyle([
            ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
            ("FONTSIZE", (0,0), (-1,-1), 6),
            ("TOPPADDING", (0,0), (-1,-1), 2),
            ("BOTTOMPADDING", (0,0), (-1,-1), 2),
        ]))
        elements.extend([tbl, Spacer(1,12)])
elements.append(PageBreak())

# 3. Tokenizer Vocabulary
elements.append(Paragraph("3. Tokenizer Vocabulary", styles["Heading2"]))
with open(os.path.join(TOKEN_DIR, "vocab.json"), "r", encoding="utf-8") as f:
    mapping = json.load(f)
items = sorted(mapping.items(), key=lambda x: int(x[0]))

chunk_size = 50
for i in range(0, len(items), chunk_size):
    chunk = items[i:i+chunk_size]
    data = [[idx, piece] for idx, piece in chunk]
    tbl = Table(data, colWidths=[40, 200])
    tbl.setStyle(TableStyle([
        ("GRID", (0,0), (-1,-1), 0.3, colors.grey),
        ("FONTSIZE", (0,0), (-1,-1), 6),
        ("ALIGN", (0,0), (-1,-1), "LEFT"),
    ]))
    elements.extend([tbl, Spacer(1,12)])

doc.build(elements)
print(f"PDF saved to {OUTPUT_PDF}")
