import os, json, numpy as np

BASE = r"C:\junha\Git\Analog_GPT\Extracted"
OUT  = os.path.join(BASE, "tex_all")
os.makedirs(OUT, exist_ok=True)

def bm(a):
    if a.ndim == 1:
        return "\\begin{bmatrix}\n" + " & ".join(f"{v:.6f}" for v in a) + "\n\\end{bmatrix}"
    rows = " \\\\\n".join(" & ".join(f"{v:.6f}" for v in r) for r in a)
    return "\\begin{bmatrix}\n" + rows + "\n\\end{bmatrix}"

def save(a, name):
    with open(os.path.join(OUT, name + ".tex"), "w") as f:
        f.write(f"\\documentclass{{standalone}}\n\\usepackage{{amsmath}}\n\\begin{{document}}\n{bm(a)}\n\\end{{document}}\n")

for root, _, files in os.walk(BASE):
    for fn in files:
        if not fn.endswith(".npy"):
            continue
        arr = np.load(os.path.join(root, fn), mmap_mode="r")
        stem = os.path.splitext(fn)[0]
        if arr.ndim <= 2:
            save(arr, stem)
        elif arr.ndim == 3:
            for i in range(arr.shape[0]):
                save(arr[i], f"{stem}_{i}")
        elif arr.ndim == 4:
            for i in range(arr.shape[0]):
                for j in range(arr.shape[1]):
                    save(arr[i, j], f"{stem}_{i}_{j}")

with open(os.path.join(BASE, "tokenizer", "vocab.json"), encoding="utf-8") as f:
    vocab = json.load(f)
items = sorted(vocab.items(), key=lambda x: int(x[0]))
chunk = 50
table_tex = []
for i in range(0, len(items), chunk):
    rows = " \\\\\n".join(f"{idx} & {tok}" for idx, tok in items[i:i+chunk])
    table_tex.append("\\begin{tabular}{rr}\n" + rows + "\n\\end{tabular}")
with open(os.path.join(OUT, "vocab.tex"), "w", encoding="utf-8") as f:
    f.write("\\documentclass{standalone}\n\\usepackage{booktabs}\n\\begin{document}\n" +
            "\\par\\bigskip\n".join(table_tex) + "\n\\end{document}\n")
