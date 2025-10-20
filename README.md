# 🤖 Quirky: Explainable Anomaly Detection

**Quirky** is a feature-agnostic Python package designed for explainable anomaly detection. It leverages the robust **Isolation Forest (iForest)** algorithm for core detection and utilizes **SHAP (SHapley Additive exPlanations)** to provide comprehensive, visual model interpretations.

The primary goal of `Quirky` is to answer not just *what* data points are anomalous, but *why* they were flagged by the model.

## 🚀 Quick Start (User Guide)

### Installation

Install the package directly from the Git repository using `pip`:

```bash
git clone https://github.com/your-username/quirky.git
cd quirky
pip install .
```

### Basic Usage

The core of the package is the `Quirky` class, which manages data, training, scoring, and analysis in a controlled sequence.

```python
import pandas as pd
import numpy as np
from quirky import Quirky # Assuming Quirky is available after installation
import matplotlib.pyplot as plt

# 1. Define Features and Configuration
MODEL_FEATURES = ['Trans_Amount', 'IP_Count', 'Device_Type']
CONTAMINATION = 0.01  # Expected percentage of anomalies in the dataset
RANDOM_STATE = 42

# --- Prepare Dummy Data (Replace with your actual data loading) ---
# X_train should be the "clean" data subset (known normal transactions)
X_train = pd.DataFrame(np.random.rand(1000, 3) * 10, columns=MODEL_FEATURES)
# X_full includes anomalies (e.g., high Trans_Amount)
X_full = pd.concat([
    X_train, 
    pd.DataFrame({'Trans_Amount': [100, 110], 'IP_Count': [1, 2], 'Device_Type': [0.5, 0.4]})
], ignore_index=True)
# -----------------------------------------------------------------

# 2. Initialize and Load Data
detector = Quirky(
    feature_names=MODEL_FEATURES,
    cont_rate=CONTAMINATION,
    random_state=RANDOM_STATE
)

# Load full data and the clean training subset
detector.load_data(X_full=X_full, X_train=X_train) 

# 3. Train and Score
detector.train_model()
detector.calculate_scores()

# 4. Get Results
results_df = detector.get_analysis_results()
print("Top Anomalies:\n", results_df[results_df['Predicted_Anomaly'] == 1].sort_values('iForest_Score').head())

# 5. Get and Display Figures
validation_plots = detector.get_validation_figures()
shap_plots = detector.run_shap_analysis()

# Display the global feature importance bar plot
shap_plots['global_bar'].show() 
```

-----

## ⚙️ Class Reference (`Quirky`)

### Initialization

```python
Quirky(feature_names: List[str], cont_rate: float, random_state: int = 42, background_sample_size: int = 100)
```

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `feature_names` | `List[str]` | N/A | **Required.** The names of columns used for modeling. |
| `cont_rate` | `float` | N/A | The expected fraction of outliers, used by Isolation Forest for setting the internal threshold. |
| `random_state` | `int` | `42` | Seed for reproducibility. |
| `background_sample_size` | `int` | `100` | The number of samples used by SHAP's TreeExplainer. A larger sample gives better accuracy but increases computation time. |

### Core Methods

| Method | Returns | Description |
| :--- | :--- | :--- |
| `load_data(X_full, X_train)` | `None` | **Loads the datasets.** `X_full` is the entire data (for scoring), and `X_train` is the clean, non-anomalous data (for training). |
| `train_model()` | `None` | Fits the Isolation Forest model using `X_train`. |
| `calculate_scores()` | `None` | Scores `X_full` and sets the internal classification threshold based on `cont_rate`. |
| `set_threshold(value: float)` | `None` | Allows manual overriding of the anomaly score threshold, updating predictions instantly. |
| `get_analysis_results()` | `pd.DataFrame` | Returns the scores and prediction columns (`iForest_Score`, `Predicted_Anomaly`, etc.). |

### Visualization Methods

These methods return dictionaries of Matplotlib `Figure` objects, allowing the user full control over saving or displaying the plots.

| Method | Returns | Description |
| :--- | :--- | :--- |
| `get_validation_figures()` | `Dict[str, Figure]` | Returns key plots for validation: score distribution, feature distributions, and 2D scatter plots (if $N=2$ features). |
| `run_shap_analysis()` | `Dict[str, Figure]` | Calculates SHAP values and returns interpretation plots: global bar chart, beeswarm, local force plot (for the max anomaly), and dependence plots (if $N \ge 2$). |

-----

## 💻 Developer Documentation

### Repository Structure

The package follows a standard modern Python layout:

```
quirky/
├── quirky/
│   ├── __init__.py         # Imports the Quirky class for package access
│   └── quirky.py           # Core logic (The Quirky class)
├── scripts/
│   └── demo_usage.py       # Illustrative usage example
├── pyproject.toml          # Build and dependency configuration
└── README.md
```

### Extending and Contribution

We welcome contributions\! Please adhere to these guidelines:

1.  **Branching:** Fork the repository and create a new feature branch (`feat/new-plot-type`).
2.  **Testing:** All new methods must include comprehensive unit tests (though not included here, a `tests/` directory is assumed).
3.  **Data Integrity:** When modifying data handling, ensure that `X_full_df` and `X_train_df` are always kept separate and that the training data is never contaminated by the main detection process.

### Code Style Notes

  * **Type Hinting:** All public and private methods use Python type hints for clarity.
  * **Plotting Philosophy:** All internal plotting methods (`_plot_...`) **must return a `plt.Figure` object** and must **not** call `plt.show()` or `plt.close()`. This ensures the user maintains control over the Matplotlib event loop.
  * **Generalization:** Avoid hardcoding feature indices (e.g., `[0]` and `[1]`) unless explicitly required by a 2D plot, in which case the code must raise a **`NotImplementedError`** for $N>2$ datasets.
