"""
CMU Keystroke Dynamics Benchmark Runner.
Trains and compares Scaled Manhattan Baseline, One-Class SVM, and Isolation Forest
across all 51 subjects in DSL-StrongPasswordData.csv following the Killourhy & Maxion protocol.
"""

import os
import sys

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import json
import time
from typing import Dict, List, Any
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

from ml.dataset import load_cmu_dataset, get_all_subjects, get_user_splits, FEATURE_COLUMNS
from ml.models import ScaledManhattanDetector, OneClassSVMDetector, IsolationForestDetector
from ml.evaluate import compute_eer, calculate_far_frr

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")


def run_benchmark(
    data_path: str = None,
    train_reps: int = 200,
    impostors_per_other: int = 5,
    output_dir: str = RESULTS_DIR
) -> Dict[str, Any]:
    """
    Executes the full benchmark over all 51 subjects.
    """
    os.makedirs(output_dir, exist_ok=True)
    print("=" * 75)
    print("KEYRO: KEYSTROKE DYNAMICS OFFLINE ML BENCHMARK")
    print("=" * 75)

    start_time = time.time()
    df = load_cmu_dataset(data_path) if data_path else load_cmu_dataset()
    subjects = get_all_subjects(df)
    n_subjects = len(subjects)
    print(f"Loaded CMU Dataset: {len(df):,} records, {n_subjects} subjects, {len(FEATURE_COLUMNS)} timing features.")
    print(f"Protocol: {train_reps} train reps, {400 - train_reps} genuine test reps, { (n_subjects - 1) * impostors_per_other} impostor test reps per user.")
    print("-" * 75)

    model_factories = {
        "Scaled Manhattan": lambda: ScaledManhattanDetector(epsilon=1e-6, use_mad=False),
        "One-Class SVM": lambda: OneClassSVMDetector(nu=0.05, kernel="rbf", gamma="scale"),
        "Isolation Forest": lambda: IsolationForestDetector(n_estimators=100, contamination=0.05, random_state=42)
    }

    results_by_model: Dict[str, List[Dict[str, Any]]] = {name: [] for name in model_factories}
    all_scores_by_model: Dict[str, Dict[str, List[np.ndarray]]] = {
        name: {"gen": [], "imp": []} for name in model_factories
    }

    # Iterate over all subjects
    for idx, subj in enumerate(subjects, 1):
        X_train, X_test_gen, X_test_imp = get_user_splits(
            df, subj, train_reps=train_reps, impostors_per_other_user=impostors_per_other
        )

        for name, factory in model_factories.items():
            model = factory()
            model.fit(X_train)

            gen_scores = model.score_samples(X_test_gen)
            imp_scores = model.score_samples(X_test_imp)

            eval_res = compute_eer(gen_scores, imp_scores)
            eval_res["subject"] = subj
            
            # Default threshold metrics
            far_def, frr_def = calculate_far_frr(gen_scores, imp_scores, model.threshold)
            eval_res["far_default"] = far_def
            eval_res["frr_default"] = frr_def

            results_by_model[name].append(eval_res)
            all_scores_by_model[name]["gen"].append(gen_scores)
            all_scores_by_model[name]["imp"].append(imp_scores)

        if idx % 10 == 0 or idx == n_subjects:
            print(f"Evaluated [{idx}/{n_subjects}] subjects...")

    elapsed = time.time() - start_time
    print(f"\nCompleted evaluation in {elapsed:.2f} seconds.")
    print("=" * 75)

    # Aggregate metrics
    summary = {}
    print(f"{'Model':<20} | {'Mean EER (%)':<15} | {'Std EER (%)':<15} | {'Mean AUC':<12}")
    print("-" * 75)

    for name, res_list in results_by_model.items():
        eers = [r["eer"] * 100 for r in res_list]
        aucs = [r["auc"] for r in res_list]
        fars = [r["far_at_eer"] * 100 for r in res_list]
        frrs = [r["frr_at_eer"] * 100 for r in res_list]

        summary[name] = {
            "mean_eer": float(np.mean(eers)),
            "std_eer": float(np.std(eers)),
            "median_eer": float(np.median(eers)),
            "min_eer": float(np.min(eers)),
            "max_eer": float(np.max(eers)),
            "mean_auc": float(np.mean(aucs)),
            "std_auc": float(np.std(aucs)),
            "mean_far": float(np.mean(fars)),
            "mean_frr": float(np.mean(frrs)),
            "subject_results": [
                {
                    "subject": r["subject"],
                    "eer": round(r["eer"] * 100, 2),
                    "auc": round(r["auc"], 4),
                    "eer_threshold": round(r["eer_threshold"], 4)
                } for r in res_list
            ]
        }
        print(f"{name:<20} | {np.mean(eers):>6.2f}% ± {np.std(eers):<6.2f} | {np.std(eers):>6.2f}%        | {np.mean(aucs):>6.4f}")

    print("=" * 75)

    # Identify best model
    best_model_name = min(summary.keys(), key=lambda k: summary[k]["mean_eer"])
    print(f"\n>>> BEST PERFORMING MODEL: {best_model_name} (Lowest Average EER: {summary[best_model_name]['mean_eer']:.2f}%)")

    # Save summary JSON
    summary_path = os.path.join(output_dir, "benchmark_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved benchmark summary to: {summary_path}")

    # Save best model config
    best_config = {
        "best_model": best_model_name,
        "mean_eer_percent": summary[best_model_name]["mean_eer"],
        "mean_auc": summary[best_model_name]["mean_auc"],
        "parameters": {}
    }
    if best_model_name == "Scaled Manhattan":
        best_config["parameters"] = {"epsilon": 1e-6, "use_mad": False}
    elif best_model_name == "One-Class SVM":
        best_config["parameters"] = {"nu": 0.05, "kernel": "rbf", "gamma": "scale"}
    elif best_model_name == "Isolation Forest":
        best_config["parameters"] = {"n_estimators": 100, "contamination": 0.05, "random_state": 42}

    best_config_path = os.path.join(output_dir, "best_model_config.json")
    with open(best_config_path, "w", encoding="utf-8") as f:
        json.dump(best_config, f, indent=2)
    print(f"Saved best model config to: {best_config_path}")

    # Generate Plots
    generate_plots(all_scores_by_model, summary, output_dir)

    return summary


def generate_plots(
    all_scores: Dict[str, Dict[str, List[np.ndarray]]],
    summary: Dict[str, Any],
    output_dir: str
) -> None:
    """Generates ROC curves, DET curves, and a model comparison bar chart."""
    plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
    
    # -------------------------------------------------------------------------
    # 1. ROC Curves
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=300)
    colors = {"Scaled Manhattan": "#2563eb", "One-Class SVM": "#10b981", "Isolation Forest": "#f59e0b"}

    common_fpr = np.linspace(0, 1, 300)
    for model_name, score_dict in all_scores.items():
        tpr_interpolated = []
        for gen_s, imp_s in zip(score_dict["gen"], score_dict["imp"]):
            y_true = np.concatenate([np.zeros(len(gen_s)), np.ones(len(imp_s))])
            # Higher score = impostor (1)
            y_score = np.concatenate([gen_s, imp_s])
            fpr, tpr, _ = roc_curve(y_true, y_score)
            interp_tpr = np.interp(common_fpr, fpr, tpr)
            interp_tpr[0] = 0.0
            tpr_interpolated.append(interp_tpr)

        mean_tpr = np.mean(tpr_interpolated, axis=0)
        mean_tpr[-1] = 1.0
        roc_auc = auc(common_fpr, mean_tpr)
        plt.plot(
            common_fpr, mean_tpr,
            label=f"{model_name} (AUC = {roc_auc:.3f}, EER = {summary[model_name]['mean_eer']:.1f}%)",
            color=colors.get(model_name, "#4f46e5"),
            linewidth=2.5
        )

    plt.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random Chance (AUC = 0.50)")
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel("False Accept Rate (FAR / FPR)", fontsize=12, fontweight="bold")
    plt.ylabel("True Accept Rate (1 - FRR / TPR)", fontsize=12, fontweight="bold")
    plt.title("ROC Curves Comparison — Keystroke Dynamics (51 Subjects)", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="lower right", frameon=True, fontsize=10)
    plt.tight_layout()
    roc_path = os.path.join(output_dir, "roc_curves.png")
    plt.savefig(roc_path)
    plt.close()
    print(f"Generated ROC curves plot: {roc_path}")

    # -------------------------------------------------------------------------
    # 2. DET Curves (FRR vs FAR)
    # -------------------------------------------------------------------------
    plt.figure(figsize=(8, 6), dpi=300)
    for model_name, score_dict in all_scores.items():
        far_grid = np.logspace(-3, 0, 300)
        frr_interpolated = []
        for gen_s, imp_s in zip(score_dict["gen"], score_dict["imp"]):
            # Evaluate across sweep of thresholds
            all_s = np.concatenate([gen_s, imp_s])
            thresh_sweep = np.linspace(np.min(all_s), np.max(all_s), 400)
            far_sweep = np.mean(imp_s[:, None] < thresh_sweep[None, :], axis=0)
            frr_sweep = np.mean(gen_s[:, None] >= thresh_sweep[None, :], axis=0)

            # Sort by FAR for interpolation
            sort_idx = np.argsort(far_sweep)
            interp_frr = np.interp(far_grid, far_sweep[sort_idx], frr_sweep[sort_idx])
            frr_interpolated.append(interp_frr)

        mean_frr = np.mean(frr_interpolated, axis=0)
        plt.plot(
            far_grid * 100, mean_frr * 100,
            label=f"{model_name} (EER = {summary[model_name]['mean_eer']:.1f}%)",
            color=colors.get(model_name, "#4f46e5"),
            linewidth=2.5
        )

    # Plot diagonal EER line
    diag = np.linspace(0.1, 100, 200)
    plt.plot(diag, diag, "r:", alpha=0.6, label="EER Line (FAR = FRR)")

    plt.xscale("log")
    plt.yscale("log")
    plt.xlim([0.1, 100])
    plt.ylim([0.1, 100])
    plt.xlabel("False Accept Rate (FAR %)", fontsize=12, fontweight="bold")
    plt.ylabel("False Reject Rate (FRR %)", fontsize=12, fontweight="bold")
    plt.title("Detection Error Tradeoff (DET) Curves", fontsize=14, fontweight="bold", pad=12)
    plt.legend(loc="upper right", frameon=True, fontsize=10)
    plt.tight_layout()
    det_path = os.path.join(output_dir, "det_curves.png")
    plt.savefig(det_path)
    plt.close()
    print(f"Generated DET curves plot: {det_path}")

    # -------------------------------------------------------------------------
    # 3. Model Comparison Bar Chart
    # -------------------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    model_names = list(summary.keys())
    eer_means = [summary[m]["mean_eer"] for m in model_names]
    eer_stds = [summary[m]["std_eer"] for m in model_names]
    bar_colors = [colors.get(m, "#4f46e5") for m in model_names]

    x = np.arange(len(model_names))
    bars = plt.bar(x, eer_means, yerr=eer_stds, capsize=8, color=bar_colors, alpha=0.85, width=0.55, edgecolor="black")

    for bar, mean_val in zip(bars, eer_means):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1.2,
            f"{mean_val:.2f}%",
            ha="center", va="bottom", fontsize=11, fontweight="bold"
        )

    plt.xticks(x, model_names, fontsize=11, fontweight="bold")
    plt.ylabel("Equal Error Rate (EER %)", fontsize=12, fontweight="bold")
    plt.title("Model Comparison: Average EER ± 1 Std across 51 Users", fontsize=14, fontweight="bold", pad=14)
    plt.ylim([0, max(eer_means) + max(eer_stds) + 5])
    plt.tight_layout()
    bar_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(bar_path)
    plt.close()
    print(f"Generated comparison bar chart: {bar_path}")


if __name__ == "__main__":
    run_benchmark()
