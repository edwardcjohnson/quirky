import pandas as pd
import numpy as np
from sklearn.datasets import make_blobs
from quirky.quirky import Quirky
import matplotlib.pyplot as plt
from typing import List, Tuple

# --- Data Generation Function (Specific to this demo) ---
def generate_demo_data(feature_names: List[str], n_normal: int = 1000, n_outliers: int = 50, random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates the specific 2-feature dataset for the demo, providing separate full and training data.
    
    Returns:
        X_full_df (pd.DataFrame): The full dataset including anomalies.
        X_train_df (pd.DataFrame): The clean, normal subset for training.
    """
    np.random.seed(random_state)
    
    # 1. Normal Data (Used for clean training set)
    # n_features must be 2 for this demo's custom anomaly generation and 2D plotting.
    X_train_array, _ = make_blobs(n_samples=n_normal, n_features=2, centers=1, cluster_std=2.0, random_state=random_state)
    X_train_df = pd.DataFrame(X_train_array, columns=feature_names)
    
    # 2. Anomalies (F1 dominant, F2 slight)
    X_outliers_F1 = np.random.uniform(low=20, high=30, size=(n_outliers, 1)) 
    X_outliers_F2 = np.random.uniform(low=7, high=10, size=(n_outliers, 1))  
    X_outliers_array = np.hstack([X_outliers_F1, X_outliers_F2])
    X_outliers_df = pd.DataFrame(X_outliers_array, columns=feature_names)

    # 3. Full Data (Combining Normal + Anomalies)
    X_full_df = pd.concat([X_train_df, X_outliers_df], ignore_index=True)
    
    # Add 'True_Anomaly' for visualization/validation purposes in the full set
    X_full_df['True_Anomaly'] = np.array([0] * n_normal + [1] * n_outliers)
    X_full_df['True_Anomaly_Label'] = X_full_df['True_Anomaly'].map({0: 'Normal', 1: 'Anomaly'})

    return X_full_df, X_train_df


# --- Main Demo Workflow ---
if __name__ == "__main__":
    
    print("--- Quirky Explainable Anomaly Detector Demo ---")
    
    # 1. Configuration
    # NOTE: The feature names must be used consistently.
    MODEL_FEATURES = ['Transaction_Amount', 'Login_Duration']
    CONTAMINATION_RATE = 0.05
    RANDOM_STATE = 42

    # 2. Generate and Load Data
    full_data, train_data = generate_demo_data(MODEL_FEATURES, random_state=RANDOM_STATE)
    
    # 3. Instantiate and Configure the Detector
    detector = Quirky(
        feature_names=MODEL_FEATURES,
        cont_rate=CONTAMINATION_RATE, 
        random_state=RANDOM_STATE
    )

    # 4. Load Data (Explicitly separating X_full and X_train)
    detector.load_data(X_full=full_data, X_train=train_data)
    
    # 5. Train and Score
    detector.train_model()
    detector.calculate_scores()
    
    # 6. Get Results
    results_df = detector.get_analysis_results()
    
    print("\n--- Final Anomaly Score Results (Top 5 Predicted Anomalies) ---")
    # Anomalies have the lowest (most negative) scores
    top_anomalies = results_df[results_df['Predicted_Anomaly'] == 1].sort_values('iForest_Score').head()
    print(top_anomalies)

    # 7. Generate Figures
    validation_figures = detector.get_validation_figures()
    analysis_figures = detector.run_shap_analysis()
    
    # Combine all figures for display
    all_figures = {**validation_figures, **analysis_figures}
    
    print(f"\nDisplaying {len(all_figures)} generated figures (close window to continue):")
    
    # 8. Display Figures (The user handles plt.show() outside the class)
    # The figures are sequentially displayed until the user closes them all.
    for name, fig in all_figures.items():
        print(f"  > Showing: {name}")
        plt.figure(fig.number)
        plt.show()
