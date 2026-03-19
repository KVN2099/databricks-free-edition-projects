## MNIST classifier

**MNIST** is the classic handwritten-digit dataset: small grayscale images (28×28), ten classes, ideal for proving that **Keras/TensorFlow**, **GPU or CPU training**, and **MLflow** tracking work in your Databricks environment before you tackle larger vision or NLP projects.

### What this demo shows

- **Preprocessing:** scaling pixel values, reshaping tensors, and splitting train/validation (and optionally test) sets.
- **Model definition:** a small feed-forward or CNN architecture (as implemented in the training notebook).
- **Experiment tracking:** logging **loss**, **accuracy**, hyperparameters, and optionally the trained artifact with **MLflow** so runs are comparable in the UI.

Flow:

```text
MNIST load → normalize/split → Keras model.fit → MLflow log → (optional) register model
```

### Run order

| Step | Notebook | Purpose |
| ---- | -------- | ------- |
| 1 | `1. Data Preprocessing.ipynb` | Obtain MNIST (e.g. built-in Keras load or files), apply normalization, persist or pass tensors according to the notebook’s pattern. |
| 2 | `2. Model Training.ipynb` | Define the network, compile, train, evaluate; integrate MLflow autolog or explicit `mlflow.keras` / `mlflow.tensorflow` logging as written. |

### Prerequisites

- **Runtime:** Databricks runtime with TensorFlow, or install from `requirements.txt`:
  - `keras`, `tensorflow`
  - `numpy` within the range pinned (TensorFlow is sensitive to NumPy versions)
  - `mlflow` for tracking
- **Storage:** if notebooks write to **Unity Catalog** or **DBFS**, ensure your user can create or overwrite those paths.
- **Compute:** CPU is often enough for MNIST; GPU shortens epochs if available.

### Configuration

- Set any **widgets** or **constants** at the top for **experiment name**, **catalog.schema** (if saving tables), and **checkpoint paths**.
- Prefer a dedicated MLflow **experiment** per project so runs do not mix with unrelated work.

### Troubleshooting

- **TensorFlow / NumPy mismatch:** use the versions in `requirements.txt` or the runtime’s bundled pair; mixing pip upgrades can break `import tensorflow`.
- **MLflow run not showing metrics:** confirm the experiment URI (workspace vs UC-backed) and that the training notebook completed without aborting before log calls.
- **Slow on CPU:** reduce epochs for smoke tests; MNIST converges quickly even with small budgets.
