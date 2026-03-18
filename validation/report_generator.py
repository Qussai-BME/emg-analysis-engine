"""
report_generator.py  — v4

Fixes vs v3:
  ✓ UTF-8 encoding on all file writes  (fixes ??? instead of ±)
  ✓ Feature table capped at 20 features (236-column table is unreadable)
  ✓ Shows first channel's features only in table  (cleaner summary)
"""

import os
import json
import sys
import traceback
from datetime import datetime

try:
    import markdown
    HAS_MARKDOWN = True
except ImportError:
    HAS_MARKDOWN = False

from validation.metrics import plot_confusion_matrix


def generate_report(dataset_name, config, results, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # ── JSON ──────────────────────────────────────────────────────
    json_path = os.path.join(output_dir, f"{dataset_name}_results.json")
    print(f"Saving JSON → {json_path}", flush=True)
    try:
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print("JSON saved.", flush=True)
    except Exception as e:
        print(f"ERROR saving JSON: {e}", flush=True)
        traceback.print_exc()
        return

    # ── Markdown ─────────────────────────────────────────────────
    try:
        md = []
        md.append(f"# Validation Report: {dataset_name}")
        md.append(f"*Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}*\n")

        md.append("## Dataset Overview")
        md.append(f"- Subjects: {results.get('n_subjects', 'N/A')}")
        md.append(f"- Channels: {results.get('n_channels', 'N/A')}")
        md.append(f"- Sampling rate: {results.get('sampling_rate', 'N/A')} Hz")
        md.append(f"- Movements: {results.get('n_movements', 'N/A')}\n")

        md.append("## Processing Parameters")
        for k, v in config.items():
            md.append(f"- **{k}**: {v}")
        md.append("")

        # Feature stats — cap at first 20 features for readability
        md.append("## Feature Statistics (first 20 features, Mean +/- Std)")
        stats = results.get('feature_stats', {})
        if stats:
            classes   = sorted(stats.keys(), key=lambda x: int(x) if x.lstrip('-').isdigit() else x)
            # Feature names from first class
            all_feats = list(stats[classes[0]].keys())
            feat_cols = all_feats[:20]   # cap at 20

            header = "| Movement | " + " | ".join(feat_cols) + " |"
            sep    = "|---|" + "|".join(["---"] * len(feat_cols)) + "|"
            md.append(header)
            md.append(sep)
            for cls in classes:
                row = f"| {cls} |"
                for fn in feat_cols:
                    if fn in stats[cls]:
                        mean, std = stats[cls][fn]
                        row += f" {mean:.4f} +/- {std:.4f} |"
                    else:
                        row += " N/A |"
                md.append(row)
        md.append("")

        # Classification
        md.append("## Classification Results")
        clf_res = results.get('classification')
        if clf_res is not None:
            acc, std_acc, cm = clf_res
            md.append(f"- **Classifier**: {config.get('classifier', 'see config')}")
            md.append(f"- **Cross-validation**: Leave-One-Subject-Out (LOSO)")
            md.append(f"- **PCA**: inside each fold (no data leakage)")
            md.append(f"- **Accuracy**: {acc:.2%} +/- {std_acc:.2%}")
            cm_path = os.path.join(output_dir, f"{dataset_name}_cm.png")
            class_names = results.get('class_names', [])
            plot_confusion_matrix(cm, class_names, cm_path)
            md.append(f"![Confusion Matrix]({os.path.basename(cm_path)})")
        else:
            md.append("*No classification performed.*")
        md.append("")

        # Issues
        issues = results.get('issues', [])
        md.append("## Issues")
        if issues:
            for issue in issues:
                md.append(f"- {issue}")
        else:
            md.append("None.")

        md_text  = "\n".join(md)
        md_path  = os.path.join(output_dir, f"{dataset_name}_report.md")
        print(f"Writing Markdown → {md_path}", flush=True)
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_text)

        # HTML
        if HAS_MARKDOWN:
            html_body = markdown.markdown(md_text, extensions=['tables'])
        else:
            # Fallback: wrap pre-formatted
            html_body = f"<pre>{md_text}</pre>"

        html_path = os.path.join(output_dir, f"{dataset_name}_report.html")
        print(f"Writing HTML → {html_path}", flush=True)
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(
                f"<!DOCTYPE html><html><head>"
                f"<meta charset='utf-8'>"
                f"<title>{dataset_name} Report</title>"
                f"<style>body{{font-family:sans-serif;max-width:1200px;margin:auto;padding:2em}}"
                f"table{{border-collapse:collapse;width:100%}}"
                f"th,td{{border:1px solid #ccc;padding:4px 8px;font-size:0.8em}}</style>"
                f"</head><body>{html_body}</body></html>"
            )

        print(f"Report complete: {md_path}, {html_path}, {json_path}", flush=True)

    except Exception as e:
        print(f"ERROR generating report: {e}", flush=True)
        traceback.print_exc()