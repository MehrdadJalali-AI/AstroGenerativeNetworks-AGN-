from __future__ import annotations

MODEL_NAME_MAP = {
    "rc_agn": "AGN",
    "agn": "AGN",
    "standard_agn": "Unconditioned + cosine",
    "unconditioned_cosine": "Unconditioned + cosine",
    "fukushima_yamanishi_gca": "Fukushima-Yamanishi-style GCA",
    "knn_raw": "kNN/raw",
    "knn_raw_plus_structural": "kNN/raw+structural",
    "preferential": "Preferential",
    "random": "Random",
    "vgae": "VGAE/GAE",
    "gae": "VGAE/GAE",
    "sage_attach": "GraphSAGE + learned attachment",
    "observed_incomplete": "Observed incomplete",
    "all": "All models",
}

MAIN_MODEL_ORDER = [
    "AGN",
    "Unconditioned + cosine",
    "Fukushima-Yamanishi-style GCA",
    "kNN/raw",
    "Preferential",
    "Random",
    "VGAE/GAE",
]


def display_model_name(name: object) -> object:
    if not isinstance(name, str):
        return name
    return MODEL_NAME_MAP.get(name, name)


def apply_model_display_names(df):
    out = df.copy()
    if "model" in out.columns:
        out["model"] = out["model"].map(display_model_name)
    if "baseline" in out.columns:
        out["baseline"] = out["baseline"].map(display_model_name)
    return out

