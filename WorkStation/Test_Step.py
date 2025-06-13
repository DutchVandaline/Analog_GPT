import math
import torch
from sklearn.metrics import f1_score
from typing import Tuple

def test_step(
        model,
        dataloader,
        loss_fn,
        device,
        use_amp: bool = False,
) -> Tuple[float, float, float]:
    model.eval()

    total_ce_sum = 0.0
    total_tok = 0
    correct_tok = 0
    all_lbls, all_preds = [], []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            # StreamingDataset의 배치 형식 처리
            inputs = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            with torch.amp.autocast(enabled=use_amp, device_type=device.type):
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out

            logits_flat = logits.view(-1, logits.size(-1))
            labels_flat = labels.view(-1)

            # 손실 계산 (pad_id=0에 맞게 ignore_index=0)
            batch_ce = loss_fn(logits_flat, labels_flat)
            num_tok = (labels_flat != 0).sum().item()

            # 유효 토큰 없음 경고
            if num_tok == 0:
                print(f"Warning: Batch {batch_idx} has no valid tokens (all labels are pad_id=0)")
                continue

            batch_ce_sum = batch_ce.item() * num_tok

            total_ce_sum += batch_ce_sum
            total_tok += num_tok

            preds_flat = logits_flat.argmax(dim=1)
            mask = labels_flat != 0
            preds_masked = preds_flat[mask]
            labels_masked = labels_flat[mask]

            mask = labels_flat != 0
            correct_tok += (preds_flat[mask] == labels_flat[mask]).sum().item()
            all_lbls.extend(labels_masked.cpu().tolist())
            all_preds.extend(preds_masked.cpu().tolist())

            # 디버깅 로그
            # if batch_idx % 10 == 0:  # 10 배치마다 출력
            #     valid_ratio = num_tok / labels_flat.numel()
            #     print(f"Validation Batch {batch_idx}:")
            #     print(f"  Batch CE: {batch_ce.item():.4f}, Num Tokens: {num_tok} (Valid Ratio: {valid_ratio:.4f})")
            #     print(f"  Logits Sample: {logits_flat[0, :5].detach().cpu().tolist()}")
            #     print(f"  Predictions: {preds_masked[:5].detach().cpu().tolist()}")
            #     print(f"  Labels: {labels_masked[:5].detach().cpu().tolist()}")
            #     if batch_ce.isnan() or batch_ce.isinf():
            #         print("Warning: Batch CE is NaN or Inf")

    if total_tok == 0:
        print("Warning: No valid tokens processed in test step")
        return 1.0, 0.0, 0.0  # Perplexity=1.0, Acc=0.0, F1=0.0

    avg_ce = total_ce_sum / total_tok
    ppl = math.exp(avg_ce)
    acc = correct_tok / total_tok
    f1 = f1_score(all_lbls, all_preds, average="weighted") if total_tok > 0 else 0.0

    print(f"Test Step Summary: Avg CE = {avg_ce:.4f}, PPL = {ppl:.4f}, Acc = {acc:.4f}, F1 = {f1:.4f}")
    return ppl, acc, f1