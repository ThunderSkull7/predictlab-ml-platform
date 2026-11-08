"""Generate high-resolution 1200x630 GitHub social preview card for PredictLab."""

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def generate_social_preview() -> None:
    """Render a 1200x630 branded social preview image aligned with the dark theme."""
    out_reports = PROJECT_ROOT / "reports" / "figures" / "social_preview.png"
    out_docs = PROJECT_ROOT / "docs" / "assets" / "social_preview.png"
    out_reports.parent.mkdir(parents=True, exist_ok=True)
    out_docs.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(12, 6.3), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor("#0B0F19")
    ax.axis("off")

    # Outer border card
    rect_bg = patches.FancyBboxPatch(
        (0.03, 0.04),
        0.94,
        0.92,
        boxstyle="round,pad=0.02,rounding_size=0.03",
        linewidth=1.5,
        edgecolor="#1E293B",
        facecolor="#0F172A",
        transform=ax.transAxes,
    )
    ax.add_patch(rect_bg)

    # Vector Icon Box for PredictLab
    icon_box = patches.FancyBboxPatch(
        (0.08, 0.81),
        0.055,
        0.085,
        boxstyle="round,pad=0.01,rounding_size=0.015",
        linewidth=1,
        edgecolor="#38BDF8",
        facecolor="#1E293B",
        transform=ax.transAxes,
    )
    ax.add_patch(icon_box)

    # Trend Line inside Icon Box
    pts_x = np.array([0.088, 0.098, 0.108, 0.124])
    pts_y = np.array([0.835, 0.855, 0.845, 0.880])
    ax.plot(pts_x, pts_y, color="#38BDF8", linewidth=2.5, transform=ax.transAxes)
    ax.scatter(pts_x, pts_y, color="#38BDF8", s=18, zorder=5, transform=ax.transAxes)

    # Brand Title
    ax.text(
        0.15,
        0.855,
        "PredictLab",
        transform=ax.transAxes,
        fontsize=32,
        fontweight="bold",
        color="#38BDF8",
        va="center",
        parse_math=False,
    )
    ax.text(
        0.38,
        0.855,
        "•  AI/ML Platform",
        transform=ax.transAxes,
        fontsize=20,
        color="#64748B",
        va="center",
        parse_math=False,
    )

    # Subheadings
    ax.text(
        0.08,
        0.74,
        "Production-Grade Predictive Analytics Platform",
        transform=ax.transAxes,
        fontsize=21,
        fontweight="bold",
        color="#F8FAFC",
        va="center",
        parse_math=False,
    )
    ax.text(
        0.08,
        0.67,
        "Retrospective Customer Churn Classification, Cost-Optimal Thresholding & Explainable Inference",
        transform=ax.transAxes,
        fontsize=12.5,
        color="#94A3B8",
        va="center",
        parse_math=False,
    )

    # Metrics Cards Container
    metrics = [
        ("Primary Metric (PR-AUC)", "0.6684 ± 0.02", "#34D399"),
        ("ROC-AUC Score", "0.8495 ± 0.01", "#38BDF8"),
        ("Illustrative Policy Threshold", "τ* = 0.19 (Decision Policy)", "#F59E0B"),
        ("Brier Score", "0.1335 (Sigmoid)", "#A78BFA"),
    ]

    for idx, (title, val, col) in enumerate(metrics):
        x = 0.08 + idx * 0.215
        card = patches.FancyBboxPatch(
            (x, 0.38),
            0.195,
            0.22,
            boxstyle="round,pad=0.015,rounding_size=0.02",
            linewidth=1,
            edgecolor="#334155",
            facecolor="#1E293B",
            transform=ax.transAxes,
        )
        ax.add_patch(card)
        ax.text(
            x + 0.015,
            0.54,
            title,
            transform=ax.transAxes,
            fontsize=9.5,
            color="#94A3B8",
            fontweight="bold",
            va="center",
            parse_math=False,
        )
        ax.text(
            x + 0.015,
            0.45,
            val,
            transform=ax.transAxes,
            fontsize=14.5,
            color=col,
            fontweight="bold",
            va="center",
            parse_math=False,
        )

    # Feature Pills
    pills = [
        ("Python 3.12 64-bit", "#38BDF8"),
        ("Scikit-Learn 1.6.1 Pipelines", "#34D399"),
        ("Platt Sigmoid Calibration", "#A78BFA"),
        ("TreeSHAP Log-Odds Attributions", "#F59E0B"),
        ("Interactive Streamlit 1.67 UI", "#F43F5E"),
        ("Fail-Closed SHA-256 Engine", "#10B981"),
    ]

    for idx, (pill, dot_col) in enumerate(pills):
        px = 0.08 + (idx % 3) * 0.285
        py = 0.25 if idx < 3 else 0.16
        p_box = patches.FancyBboxPatch(
            (px, py - 0.025),
            0.265,
            0.065,
            boxstyle="round,pad=0.01,rounding_size=0.015",
            linewidth=0.8,
            edgecolor="#334155",
            facecolor="#111827",
            transform=ax.transAxes,
        )
        ax.add_patch(p_box)
        ax.scatter([px + 0.022], [py + 0.007], color=dot_col, s=28, transform=ax.transAxes)
        ax.text(
            px + 0.038,
            py + 0.007,
            pill,
            transform=ax.transAxes,
            fontsize=10,
            color="#E2E8F0",
            va="center",
            parse_math=False,
        )

    # Footer note
    ax.text(
        0.08,
        0.07,
        "IBM Telco Customer Churn Snapshot • 47 Baseline Predictors • Illustrative Cost Scenario • MIT License",
        transform=ax.transAxes,
        fontsize=9.5,
        color="#64748B",
        va="center",
        parse_math=False,
    )

    plt.savefig(out_reports, dpi=100, facecolor="#0B0F19")
    plt.savefig(out_docs, dpi=100, facecolor="#0B0F19")
    plt.close()

    im = Image.open(out_reports)
    print(f"Successfully generated social preview: {out_reports} ({im.size[0]}x{im.size[1]} px)")


if __name__ == "__main__":
    generate_social_preview()
