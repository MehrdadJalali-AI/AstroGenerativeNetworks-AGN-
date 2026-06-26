"""
Matplotlib / seaborn styling for IEEE two-column, publication-ready figures.
"""

import matplotlib as mpl
import matplotlib.pyplot as plt


def apply_paper_matplotlib_style():
    """Minimal, high-contrast, readable defaults (vector-friendly)."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.titleweight": "600",
            "axes.labelweight": "500",
            "axes.edgecolor": "#222222",
            "axes.linewidth": 0.9,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "grid.linewidth": 0.6,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#111111",
            "axes.labelcolor": "#111111",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    mpl.rcParams["axes.prop_cycle"] = mpl.cycler(
        color=["#0B7285", "#E8590C", "#5F3DC4", "#2B8A3E", "#C92A2A"]
    )
