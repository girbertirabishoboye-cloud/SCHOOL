# Smart Classroom Energy System

An ML model predicts whether a classroom is **occupied** from sensor readings
(light, temperature, humidity, CO2, hour). A transparent rule then recommends
the lighting action: **ON = occupied AND dark, otherwise OFF.**

## Folder structure

```
Smart-Classroom-Energy-System/
├── app.py                  Streamlit dashboard
├── core.py                 Prediction + lighting rule + energy maths (testable)
├── train_model.py          Trains on the Kaggle CSV, writes model + evaluation reports
├── requirements.txt
├── models/
│   └── smart_classroom_final_model.joblib
├── data/                   Put the Kaggle CSV here (see data/README.md)
├── reports/                Created by train_model.py (metrics, confusion matrix ...)
├── tests/test_core.py
└── docs/SUBMISSION_CHECKLIST.md   Lecturer feedback -> what to do, slide text, reflections
```

## Run locally

```
pip install -r requirements.txt
python -m streamlit run app.py
```

(From Jupyter on Windows: `python -m streamlit run "D:\my project machine learing\app.py"`
after copying these files into that folder.)

## Why the old error happened

The old `app.py` only searched for `smart_classroom_decision_tree.joblib`, and it
expected a model that answers ON/OFF from `lux, motion, hour`. Your
`smart_classroom_final_model.joblib` is a *different* model (inputs
`light_level, temperature, humidity, co2, hour`, output = occupied or not), so
renaming or replacing the file could not work. This version:

* looks for `models/smart_classroom_final_model.joblib` (or the project root, or any other
  non-backup `.joblib` in `models/`),
* uses the final model's real inputs, and
* turns its occupancy prediction into ON/OFF with a documented rule.

## Retrain on the dataset (needed for the lecturer's feedback)

```
python train_model.py --data data/<kaggle_file>.csv --dataset-name "Author, Title, Kaggle, URL"
```

This overwrites the model in `models/` and fills `reports/` (confusion matrix,
precision/recall for both classes, baseline, model comparison). The dashboard's
**Model Evaluation** tab then shows these numbers automatically.

Re-training also removes any scikit-learn version mismatch warnings.

## Deploying (Streamlit Community Cloud / GitHub)

Commit `app.py`, `core.py`, `requirements.txt` and the `models/` folder, and
`reports/` after training. The model path is relative, so no code changes are needed.

## Tests

```
python tests/test_core.py
```

## Limitations

* Readings are typed in manually; no real sensor or relay is connected.
* The darkness threshold is a design choice - calibrate it to your sensor.
* Random train/test splitting of time-series sensor data can be optimistic
  because neighbouring readings are similar; state this in the report.
* Lighting ON/OFF labels come from a rule applied to sensor data, not from
  observing what people actually did.
