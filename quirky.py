import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.datasets import make_blobs
import shap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Dict, Any, Union

# Configure environment and plotting styles once
sns.set_theme(style="whitegrid", palette="viridis")
# Disable JS dependency for cleaner package usage outside of notebooks
shap.initjs(disable_js=True) 


class Quirky:
    """
    A feature-agnostic class for training an Isolation Forest anomaly detector 
    and generating comprehensive SHAP-based interpretation.
    """
    
    def __init__(self, feature_names: List[str], cont_rate: float, 
                 random_state: int = 42, background_sample_size: int = 100):
        """
        Initializes the detector with core parameters.
        
        Args:
            feature_names (List[str]): The column names (features) to be used in the model.
            cont_rate (float): The expected contamination rate for the Isolation Forest.
            random_state (int): Seed for reproducibility.
            background_sample_size (int): Size of the sample used for SHAP background data 
                                          (affects speed and accuracy of explanation).
        """
        if not feature_names:
            raise ValueError("Feature names list cannot be empty.")
            
        self.feature_names = feature_names
        self.n_features = len(feature_names)
        self.cont_rate = cont_rate
        self.random_state = random_state
        self.bg_sample_size = background_sample_size
        
        # Core data/model objects
        self.X_full_df: pd.DataFrame = None
        self.X_train_df: pd.DataFrame = None
        self.model: IsolationForest = None
        self.threshold: float = None

    # --- A. Data Loading and Preparation ---

    def load_data(self, X_full: pd.DataFrame, X_train: pd.DataFrame) -> None:
        """
        Loads the full dataset and the clean training subset into the detector.
        
        Args:
            X_full (pd.DataFrame): The full dataset to be analyzed (including potential anomalies).
            X_train (pd.DataFrame): The clean, known-normal subset of data used for model training.
        """
        if not all(f in X_full.columns for f in self.feature_names):
            raise ValueError("X_full DataFrame missing one or more specified feature names.")
        if not all(f in X_train.columns for f in self.feature_names):
            raise ValueError("X_train DataFrame missing one or more specified feature names.")
        
        self.X_full_df = X_full.copy()
        self.X_train_df = X_train.copy()
        
        print(f"Data Loaded: {len(self.X_full_df)} total samples. Training samples: {len(self.X_train_df)}")

    # --- B. Model Training and Scoring Methods ---

    def train_model(self) -> None:
        """Initializes and fits the Isolation Forest model using the clean training data."""
        if self.X_train_df is None:
            raise RuntimeError("Data must be loaded via load_data() before training.")
            
        self.model = IsolationForest(
            contamination=self.cont_rate, 
            random_state=self.random_state, 
            n_estimators=100
        )
        self.model.fit(self.X_train_df[self.feature_names]) 
        print("Model Trained.")

    def calculate_scores(self) -> None:
        """Calculates scores and classification, setting the instance threshold."""
        if self.model is None:
            raise RuntimeError("Model must be trained via train_model() before scoring.")
            
        self.X_full_df['iForest_Score'] = self.model.decision_function(self.X_full_df[self.feature_names])
        self.X_full_df['Predicted_Class'] = self.model.predict(self.X_full_df[self.feature_names])
        self.X_full_df['Predicted_Anomaly'] = np.where(self.X_full_df['Predicted_Class'] == 1, 0, 1)

        # Calculate threshold
        self.threshold = self.X_full_df[self.X_full_df['Predicted_Class'] == 1]['iForest_Score'].min()
        print(f"Scores Calculated. Threshold set at {self.threshold:.4f}.")
    
    def set_threshold(self, value: float) -> None:
        """Allows user to manually override the contamination-derived score threshold."""
        self.threshold = value
        print(f"Threshold manually set to {value:.4f}.")
        # Re-apply prediction based on new threshold
        self.X_full_df['Predicted_Anomaly'] = np.where(self.X_full_df['iForest_Score'] < self.threshold, 1, 0)
        self.X_full_df['Predicted_Class'] = np.where(self.X_full_df['Predicted_Anomaly'] == 1, -1, 1)


    def get_analysis_results(self) -> pd.DataFrame:
        """Returns the full DataFrame with scores and predictions."""
        if self.threshold is None:
            raise RuntimeError("Scores must be calculated before returning results.")
        
        return self.X_full_df[['iForest_Score', 'Predicted_Class', 'Predicted_Anomaly']].copy()

    # --- C. Plotting Methods (Returns Figures) ---

    def _plot_score_distribution(self) -> plt.Figure:
        """Plots the distribution of anomaly scores with the decision threshold."""
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(self.X_full_df['iForest_Score'], bins=50, color='darkgreen', kde=False, ax=ax)
        ax.axvline(x=self.threshold, color='red', linestyle='--', linewidth=2, 
                    label=f'Threshold (Score={self.threshold:.4f})')
        ax.set_title('Isolation Forest Score Distribution')
        ax.set_xlabel('iForest Decision Function Score'); ax.set_ylabel('Count'); ax.legend()
        return fig

    def _plot_2d_validation_scatter(self, scatter_type: str = 'validation') -> plt.Figure:
        """
        Plots 2D scatter visualizations (Validation or True Outliers). Requires N=2 features.
        
        Args:
            scatter_type (str): 'validation' (by score) or 'true' (if 'True_Anomaly' exists).
        """
        if self.n_features > 2:
            raise NotImplementedError(
                "2D scatter plots are not available for datasets with > 2 features. "
                "Consider dimensionality reduction or plotting pairwise features."
            )
        
        f1, f2 = self.feature_names[0], self.feature_names[1]
        fig, ax = plt.subplots(figsize=(10, 6))

        if scatter_type == 'true' and 'True_Anomaly' in self.X_full_df.columns:
            # Plot True Outliers
            X_normal = self.X_full_df[self.X_full_df['True_Anomaly'] == 0]
            X_outliers = self.X_full_df[self.X_full_df['True_Anomaly'] == 1]
            ax.scatter(X_normal[f1], X_normal[f2], c='C0', label='Normal Data', alpha=0.6, s=20)
            ax.scatter(X_outliers[f1], X_outliers[f2], c='red', label='True Outliers', marker='X', s=80, edgecolor='black')
            ax.set_title(f'True Data Distribution: {f1} vs {f2}')
        
        elif scatter_type == 'validation':
            # Plot Model Validation by Score
            scatter = ax.scatter(self.X_full_df[f1], self.X_full_df[f2], 
                                  c=self.X_full_df['iForest_Score'], cmap='viridis', s=50, alpha=0.8,
                                  edgecolor='k', linewidth=0.5)
            predicted_outliers = self.X_full_df[self.X_full_df['Predicted_Anomaly'] == 1]
            ax.scatter(predicted_outliers[f1], predicted_outliers[f2], 
                        c='none', s=200, marker='o', edgecolor='red', linewidth=2, 
                        label=f'Predicted Anomaly ($\gamma={self.cont_rate}$)')
            fig.colorbar(scatter, ax=ax).set_label('iForest Decision Score (Lower = Anomaly)', rotation=270, labelpad=15)
            ax.set_title(f'Model Validation: {f1} vs {f2} Colored by Score')
        
        else:
            raise ValueError(f"Invalid scatter_type: {scatter_type}")

        ax.set_xlabel(f1); ax.set_ylabel(f2); ax.legend()
        return fig

    def _plot_feature_distributions(self) -> plt.Figure:
        """Plots histograms for the first two features, comparing true labels if available."""
        
        n_plots = min(self.n_features, 2)
        fig, axes = plt.subplots(1, n_plots, figsize=(8 * n_plots, 6))
        
        if n_plots == 1:
            axes = [axes] # Ensure axes is iterable
            
        hue_col = 'True_Anomaly_Label' if 'True_Anomaly_Label' in self.X_full_df.columns else None

        for i in range(n_plots):
            feature = self.feature_names[i]
            sns.histplot(data=self.X_full_df, x=feature, hue=hue_col, stat='density', 
                         common_norm=False, kde=False, bins=30, ax=axes[i], 
                         palette={'Normal': 'C0', 'Anomaly': 'red'})
            axes[i].set_title(f'{feature} Distribution')

        plt.tight_layout()
        return fig
    
    # --- D. Orchestration Methods ---

    def get_validation_figures(self) -> Dict[str, plt.Figure]:
        """
        Orchestrates the generation of core validation and EDA plots.
        Returns a dictionary of Matplotlib Figure objects.
        """
        if self.threshold is None:
            raise RuntimeError("Scores must be calculated before generating validation figures.")
        
        figures = {
            'score_distribution': self._plot_score_distribution(),
            'feature_distributions': self._plot_feature_distributions(),
        }
        
        if self.n_features == 2:
            figures['true_outlier_scatter'] = self._plot_2d_validation_scatter(scatter_type='true')
            figures['validation_score_scatter'] = self._plot_2d_validation_scatter(scatter_type='validation')
            
        print("\nValidation Figures Generated.")
        return figures

    def run_shap_analysis(self) -> Dict[str, plt.Figure]:
        """
        Orchestrates SHAP calculation and returns all interpretation figures.
        
        Returns:
            Dict[str, plt.Figure]: A dictionary mapping plot names to Matplotlib Figure objects.
        """
        if self.threshold is None:
            raise RuntimeError("Scores must be calculated before running SHAP analysis.")
            
        print("\n--- Generating SHAP Interpretation Figures ---")

        # 1. Calculate SHAP values efficiently
        background_data = self.X_train_df.sample(n=min(self.bg_sample_size, len(self.X_train_df)), 
                                                 random_state=self.random_state) 
        explainer = shap.TreeExplainer(self.model, data=background_data)
        shap_values_full = explainer.shap_values(self.X_full_df[self.feature_names])
        expected_value = explainer.expected_value
        
        # 2. Local Explanation Setup
        idx_most_anomalous = self.X_full_df['iForest_Score'].idxmin()
        most_anomalous_point = self.X_full_df.loc[[idx_most_anomalous], self.feature_names]
        shap_values_anomaly = explainer.shap_values(most_anomalous_point)
        anomaly_score_min = self.X_full_df['iForest_Score'].min()
        
        figures = {
            'force_plot': self._plot_force(
                expected_value, shap_values_anomaly, most_anomalous_point, 
                f"Local Force Plot: Most Anomalous Point ($s={anomaly_score_min:.3f}$)"
            ),
            'global_bar': self._plot_global_bar(shap_values_full),
            'beeswarm': self._plot_beeswarm(shap_values_full),
        }

        # 3. Dependence Plots (requires at least 2 features to show interaction)
        if self.n_features >= 2:
            figures['dependence_f1'] = self._plot_dependence(
                self.feature_names[0], self.feature_names[1], shap_values_full, 
                f"Dependence Plot: {self.feature_names[0]} vs. {self.feature_names[1]} Interaction (Primary)"
            )
            figures['dependence_f2'] = self._plot_dependence(
                self.feature_names[1], self.feature_names[0], shap_values_full, 
                f"Dependence Plot: {self.feature_names[1]} vs. {self.feature_names[0]} Interaction (Secondary)"
            )
        
        return figures
