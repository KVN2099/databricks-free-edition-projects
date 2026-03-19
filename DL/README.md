## Deep learning demos

These projects use **neural networks** on Databricks: image data in **Unity Catalog** (tables and volumes), **notebooks** for EDA and training, and in one case a **Databricks Asset Bundle** to version jobs, experiments, and infrastructure alongside code.

### Projects

| Project | Domain | README |
| ------- | ------ | ------ |
| [x-ray-classification](x-ray-classification/) | Binary or multi-class **chest X-ray** classification; heavier dependencies and GPU-friendly training. | [x-ray-classification/README.md](x-ray-classification/README.md) |
| [mnist-app-classifier](mnist-app-classifier/) | **MNIST** digits with **Keras/TensorFlow** and **MLflow**—lighter entry point for DL on the platform. | [mnist-app-classifier/README.md](mnist-app-classifier/README.md) |

### Skills you will exercise

- Storing **images or tensors** where Spark and UC can reference them (volumes, paths, or decoded tables depending on the notebook).
- **Experiment tracking** with MLflow (metrics, parameters, artifacts).
- Optional **IaC-style** deployment with `databricks.yml` for repeatable jobs and resource wiring (X-ray project).

### Compute guidance

- **MNIST:** often runnable on CPU; good for validating MLflow and library installs.
- **X-ray:** plan for **GPU** clusters or supported GPU Serverless where available; training time and memory scale with batch size and model depth.

Always read the project README for **run order** and **requirements**—the X-ray stack is much larger than MNIST.
