"""
Extract results from all E-SSL Figure 1 experiment logs into a single CSV.

Scans logs/ directory for run_<transform>_<inv|sen>.log files and extracts:
  - Pretrain final loss, pred_loss, pred_acc (sensitivity only)
  - ImageNet-100 best Val Acc@1 and Acc@5
  - CUB-200 best Val Acc@1 and Acc@5
  - Flowers-102 5-shot and 10-shot mean +/- CI

Usage:
    python extract_results.py
    python extract_results.py --log-dir ./logs --output results_summary.csv
"""

import argparse
import csv
import os
import re


TRANSFORMS = ["hflip", "grayscale", "rotation", "vflip", "jigsaw", "blur", "invert"]
CONDITIONS = ["inv", "sen"]
CONDITION_FULL = {"inv": "invariance", "sen": "sensitivity"}


def extract_from_log(filepath):
    """Extract all metrics from a single log file."""
    results = {
        "pretrain_final_loss": "",
        "pretrain_pred_loss": "",
        "pretrain_pred_acc": "",
        "in100_best_val_acc1": "",
        "in100_best_val_acc5": "",
        "cub200_best_val_acc1": "",
        "cub200_best_val_acc5": "",
        "flowers_5shot_mean": "",
        "flowers_5shot_ci95": "",
        "flowers_10shot_mean": "",
        "flowers_10shot_ci95": "",
    }

    if not os.path.isfile(filepath):
        return results

    with open(filepath) as f:
        lines = f.readlines()

    # Track which eval section we're in
    current_section = None

    for line in lines:
        line = line.strip()

        # Detect section transitions
        if "STEP 1:" in line or "Pretrain" in line:
            current_section = "pretrain"
        elif "STEP 2:" in line or "ImageNet-100" in line:
            current_section = "in100"
        elif "STEP 3:" in line or "CUB-200" in line:
            current_section = "cub200"
        elif "STEP 4:" in line or "Flowers" in line or "Few-shot" in line:
            current_section = "flowers"

        # Extract pretrain final epoch
        m = re.match(r"Epoch \[\d+/\d+\]\s+Loss:\s+([\d.]+)", line)
        if m and current_section == "pretrain":
            results["pretrain_final_loss"] = m.group(1)
            # Check for pred metrics on same line
            pm = re.search(r"Pred_Loss:\s+([\d.]+)\s+Pred_Acc:\s+([\d.]+)%", line)
            if pm:
                results["pretrain_pred_loss"] = pm.group(1)
                results["pretrain_pred_acc"] = pm.group(2)

        # Extract best val acc from linear eval sections
        if "*BEST*" in line:
            acc1_m = re.search(r"Val Acc@1:\s+([\d.]+)%", line)
            acc5_m = re.search(r"Val Acc@5:\s+([\d.]+)%", line)
            if acc1_m and current_section == "in100":
                results["in100_best_val_acc1"] = acc1_m.group(1)
                if acc5_m:
                    results["in100_best_val_acc5"] = acc5_m.group(1)
            elif acc1_m and current_section == "cub200":
                results["cub200_best_val_acc1"] = acc1_m.group(1)
                if acc5_m:
                    results["cub200_best_val_acc5"] = acc5_m.group(1)

        # Extract few-shot results
        shot_m = re.match(r"\s*(\d+)-shot:\s+([\d.]+)%\s+\(.*?(\d+\.?\d*)%\)", line)
        if shot_m and current_section == "flowers":
            k = shot_m.group(1)
            mean = shot_m.group(2)
            ci = shot_m.group(3)
            if k == "5":
                results["flowers_5shot_mean"] = mean
                results["flowers_5shot_ci95"] = ci
            elif k == "10":
                results["flowers_10shot_mean"] = mean
                results["flowers_10shot_ci95"] = ci

    return results


def main():
    parser = argparse.ArgumentParser(description="Extract E-SSL results to CSV")
    parser.add_argument("--log-dir", type=str, default="./logs")
    parser.add_argument("--output", type=str, default="results_summary.csv")
    args = parser.parse_args()

    fieldnames = [
        "transformation",
        "condition",
        "pretrain_final_loss",
        "pretrain_pred_loss",
        "pretrain_pred_acc",
        "in100_best_val_acc1",
        "in100_best_val_acc5",
        "cub200_best_val_acc1",
        "cub200_best_val_acc5",
        "flowers_5shot_mean",
        "flowers_5shot_ci95",
        "flowers_10shot_mean",
        "flowers_10shot_ci95",
    ]

    rows = []
    for transform in TRANSFORMS:
        for cond in CONDITIONS:
            logfile = os.path.join(args.log_dir, f"run_{transform}_{cond}.log")
            results = extract_from_log(logfile)
            row = {
                "transformation": transform,
                "condition": CONDITION_FULL[cond],
                **results,
            }
            rows.append(row)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Results written to {args.output}")
    print(f"  {len(rows)} rows ({len([r for r in rows if r['in100_best_val_acc1']])} with results)")

    # Print summary table
    print()
    print(f"{'Transform':<12} {'Condition':<12} {'IN100 Acc@1':<12} {'CUB200 Acc@1':<13} {'Flowers 5s':<12} {'Flowers 10s':<12}")
    print("-" * 73)
    for row in rows:
        in100 = row["in100_best_val_acc1"] or "-"
        cub = row["cub200_best_val_acc1"] or "-"
        f5 = row["flowers_5shot_mean"] or "-"
        f10 = row["flowers_10shot_mean"] or "-"
        print(f"{row['transformation']:<12} {row['condition']:<12} {in100:<12} {cub:<13} {f5:<12} {f10:<12}")


if __name__ == "__main__":
    main()
