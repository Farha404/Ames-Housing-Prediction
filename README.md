# Ames Housing Streamlit GUI

This is a Streamlit GUI built around the existing `Ames_Housing_Prediction.ipynb` **without modifying the notebook**.

## Project structure

```text
.
├── app.py
├── build_model.py
├── Ames_Housing_Prediction.ipynb   # unchanged
├── AmesHousing.csv          # place your original dataset here
├── requirements.txt
├── artifacts/
│   └── final_svr_model.joblib
└── model/
    ├── __init__.py
    └── project_runner.py
```

## Setup

Keep the same `AmesHousing.csv` used by the notebook in this folder.

```bash
pip install -r requirements.txt
```

Build the model artifact using the relevant original notebook cells:

```bash
python build_model.py
```

Then run the GUI:

```bash
streamlit run app.py
```

The notebook itself is not edited. `project_runner.py` reads the notebook and executes the original source of the cells needed to construct `final_svr_model`.

## Important

The model is the final SVR retrained on the notebook's training + validation data. The fitted preprocessing pipeline remains inside the model, so the Streamlit app sends the raw feature DataFrame directly to the model.

The model-performance and feature-importance pages display results already produced by the notebook; they are not presented as new evaluation experiments.
