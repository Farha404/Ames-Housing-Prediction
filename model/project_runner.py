"""Loads the existing notebook's ML logic without modifying the notebook.

The notebook remains the source of truth. This module executes only the cells needed
for inference, using their original source verbatim.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "final_modelling2.ipynb"
DATASET = ROOT / "AmesHousing.csv"
ARTIFACT = ROOT / "artifacts" / "final_svr_model.joblib"

# Exact notebook cells required to construct the final model.
MODEL_CELLS = [2, 4, 8, 10, 13, 123, 144, 146, 147, 148, 183]


def _load_notebook() -> dict[str, Any]:
    with NOTEBOOK.open("r", encoding="utf-8") as f:
        return json.load(f)


def _execute_notebook_model() -> dict[str, Any]:
    if not DATASET.exists():
        raise FileNotFoundError(
            f"Missing AmesHousing.csv. Put the dataset next to final_modelling2.ipynb: {DATASET}"
        )

    nb = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__notebook_model__"}

    # The notebook uses display() in many cells. The selected cells do not depend on
    # its return value, but defining it keeps execution robust if a selected cell does.
    namespace["display"] = lambda *args, **kwargs: None

    for index in MODEL_CELLS:
        cell = nb["cells"][index]
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)

    return namespace


def build_and_save_model() -> Path:
    """Execute the relevant original notebook cells and save the fitted final model."""
    ns = _execute_notebook_model()
    model = ns["final_svr_model"]
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, ARTIFACT)
    return ARTIFACT


def load_model():
    """Load the fitted model, building it from the unchanged notebook if necessary."""
    if not ARTIFACT.exists():
        build_and_save_model()
    return joblib.load(ARTIFACT)


def load_dataset():
    """Load the cleaned raw dataset used by the notebook for UI metadata."""
    import pandas as pd

    raw = pd.read_csv(DATASET, keep_default_na=False, na_values=[""])
    raw.columns = (
        raw.columns.str.replace(" ", "", regex=False)
        .str.replace("/", "", regex=False)
    )
    return raw


def notebook_feature_schema():
    """Return the feature taxonomy and ordinal map defined in notebook cell 10."""
    ns = _execute_notebook_schema_only()
    return (
        ns["NUMERICAL_FEATURES"],
        ns["ORDINAL_FEATURES"],
        ns["NOMINAL_FEATURES"],
        ns["ORDINAL_MAP"],
    )


def _execute_notebook_schema_only():
    if not DATASET.exists():
        raise FileNotFoundError(f"Missing dataset: {DATASET}")
    nb = _load_notebook()
    namespace: dict[str, Any] = {"__name__": "__notebook_schema__"}
    namespace["display"] = lambda *args, **kwargs: None
    for index in [2, 4, 8, 10]:
        source = "".join(nb["cells"][index].get("source", []))
        exec(compile(source, f"{NOTEBOOK.name}:cell-{index}", "exec"), namespace)
    return namespace
