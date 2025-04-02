import numpy as np
import pandas as pd
import os
from typing import List, Dict, Any, Set, Tuple

# Machine Learning Libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN, KMeans
from sklearn.ensemble import IsolationForest
from scipy import stats

class AdvancedOutlierDetector:
    """
    Comprehensive outlier detection system with multiple strategies
    
    Supports various outlier detection techniques:
    1. Statistical Z-Score Method
    2. Interquartile Range (IQR) Method
    3. Principal Component Analysis (PCA)
    4. DBSCAN Clustering
    5. Isolation Forest
    6. K-Means Clustering
    """
    
    @staticmethod
    def detect_outliers_zscore(features: np.ndarray, 
                                threshold: float = 3.0) -> Set[int]:
        """
        Detect outliers using Z-Score method
        
        Args:
            features (np.ndarray): Feature matrix
            threshold (float): Z-score threshold for outlier detection
        
        Returns:
            Set[int]: Indices of outlier frames
        """
        # Z-score calculation for each feature
        z_scores = np.abs(stats.zscore(features, axis=0))
        
        # Outliers are points with any feature Z-score above threshold
        outlier_mask = np.any(z_scores > threshold, axis=1)
        
        # Convert numpy indices to standard Python int to ensure JSON serializability
        return set(int(idx) for idx in np.where(outlier_mask)[0])
    
    @staticmethod
    def detect_outliers_iqr(features: np.ndarray, 
                             multiplier: float = 1.5) -> Set[int]:
        """
        Detect outliers using Interquartile Range (IQR) method
        
        Args:
            features (np.ndarray): Feature matrix
            multiplier (float): IQR multiplier for outlier detection
        
        Returns:
            Set[int]: Indices of outlier frames
        """
        outliers = set()
        
        for feature_col in features.T:
            Q1 = np.percentile(feature_col, 25)
            Q3 = np.percentile(feature_col, 75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - multiplier * IQR
            upper_bound = Q3 + multiplier * IQR
            
            column_outliers = np.where(
                (feature_col < lower_bound) | (feature_col > upper_bound)
            )[0]
            
            # Convert numpy indices to standard Python int
            outliers.update(int(idx) for idx in column_outliers)
        
        return outliers
    
    @staticmethod
    def detect_outliers_pca(features: np.ndarray, 
                             n_components: int = 2, 
                             variance_threshold: float = 0.95) -> Set[int]:
        """
        Detect outliers using Principal Component Analysis
        
        Args:
            features (np.ndarray): Feature matrix
            n_components (int): Number of principal components
            variance_threshold (float): Variance explained threshold
        
        Returns:
            Set[int]: Indices of outlier frames
        """
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # PCA transformation
        pca = PCA(n_components=n_components)
        pca_features = pca.fit_transform(features_scaled)
        
        # Reconstruction error
        reconstructed = pca.inverse_transform(pca_features)
        reconstruction_errors = np.mean(np.square(features_scaled - reconstructed), axis=1)
        
        # Threshold based on chi-square distribution
        threshold = np.percentile(reconstruction_errors, 95)
        
        # Convert numpy indices to standard Python int
        return set(int(idx) for idx in np.where(reconstruction_errors > threshold)[0])
    
    @staticmethod
    def detect_outliers_dbscan(features: np.ndarray, 
                                eps: float = 0.5, 
                                min_samples: int = 5) -> Set[int]:
        """
        Detect outliers using DBSCAN clustering
        
        Args:
            features (np.ndarray): Feature matrix
            eps (float): Maximum distance between two samples
            min_samples (int): Minimum number of samples in a neighborhood
        
        Returns:
            Set[int]: Indices of outlier frames
        """
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(features_scaled)
        
        # Outliers are labeled as -1, convert to standard Python int
        return set(int(idx) for idx in np.where(labels == -1)[0])
    
    @staticmethod
    def detect_outliers_isolation_forest(features: np.ndarray, 
                                         contamination: float = 0.1) -> Set[int]:
        """
        Detect outliers using Isolation Forest
        
        Args:
            features (np.ndarray): Feature matrix
            contamination (float): Expected proportion of outliers
        
        Returns:
            Set[int]: Indices of outlier frames
        """
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
        
        # Isolation Forest
        iso_forest = IsolationForest(
            contamination=contamination, 
            random_state=42
        )
        outlier_labels = iso_forest.fit_predict(features_scaled)
        
        # Outliers are labeled as -1, convert to standard Python int
        return set(int(idx) for idx in np.where(outlier_labels == -1)[0])
    
    @classmethod
    def detect_outliers_ensemble(cls, 
                             features: np.ndarray, 
                             methods: List[str] = None,
                             output_dir: str = 'outlier_analysis',
                             visualize: bool = True) -> Set[int]:
        """
        Ensemble outlier detection using multiple methods
        
        Args:
            features (np.ndarray): Feature matrix
            methods (List[str]): List of outlier detection methods to use
        
        Returns:
            Set[int]: Indices of outlier frames
        """
        # Default methods if not specified
        if methods is None:
            methods = [
                'zscore', 
                'iqr', 
                'pca', 
                'dbscan', 
                'isolation_forest'
            ]
        
        # Mapping of method names to detection functions
        detection_methods = {
            'zscore': cls.detect_outliers_zscore,
            'iqr': cls.detect_outliers_iqr,
            'pca': cls.detect_outliers_pca,
            'dbscan': cls.detect_outliers_dbscan,
            'isolation_forest': cls.detect_outliers_isolation_forest
        }
        
        # Collect outliers from multiple methods
        all_outliers = set()
        method_outliers = {}
        
        for method_name in methods:
            if method_name not in detection_methods:
                print(f"[WARNING] Unknown outlier detection method: {method_name}")
                continue
            
            method_func = detection_methods[method_name]
            outliers = method_func(features)
            
            method_outliers[method_name] = outliers
            all_outliers.update(outliers)
        
        # Toujours analyser et visualiser les résultats
        cls.analyze_outliers(features, all_outliers, output_dir)
    
        if visualize:
            cls.visualize_outliers(features, all_outliers, os.path.join(output_dir, 'visualizations'))
        
        return all_outliers
    
    @staticmethod
    def analyze_outliers(features: np.ndarray, 
                         outlier_indices: Set[int], 
                         output_dir: str = 'outlier_analysis') -> Dict[str, Any]:
        """
        Comprehensive analysis of detected outliers
        
        Args:
            features (np.ndarray): Feature matrix
            outlier_indices (Set[int]): Indices of outlier frames
            output_dir (str): Directory to save analysis results
        
        Returns:
            Dict[str, Any]: Outlier analysis metrics
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Basic statistics
        total_frames = features.shape[0]
        outlier_count = len(outlier_indices)
        
        # Convert to list for indexing and ensure they are Python integers
        outlier_indices_list = [int(i) for i in outlier_indices]
        
        # Feature distribution for outliers vs non-outliers
        if outlier_count > 0:
            outlier_features = features[outlier_indices_list]
            non_outlier_indices = list(set(range(total_frames)) - set(outlier_indices_list))
            non_outlier_features = features[non_outlier_indices] if non_outlier_indices else np.array([])
            
            # Compute statistics only if there are enough samples
            outlier_mean = np.mean(outlier_features, axis=0) if outlier_features.size > 0 else np.zeros(features.shape[1])
            outlier_std = np.std(outlier_features, axis=0) if outlier_features.size > 0 else np.zeros(features.shape[1])
            non_outlier_mean = np.mean(non_outlier_features, axis=0) if non_outlier_features.size > 0 else np.zeros(features.shape[1])
            non_outlier_std = np.std(non_outlier_features, axis=0) if non_outlier_features.size > 0 else np.zeros(features.shape[1])
        else:
            # No outliers found
            outlier_mean = np.zeros(features.shape[1])
            outlier_std = np.zeros(features.shape[1])
            non_outlier_mean = np.mean(features, axis=0)
            non_outlier_std = np.std(features, axis=0)
        
        # Convert numpy types to Python native types for JSON serialization
        outlier_stats = {
            'total_frames': int(total_frames),
            'outlier_count': int(outlier_count),
            'outlier_percentage': float(outlier_count / total_frames * 100) if total_frames > 0 else 0.0,
            'outlier_mean': outlier_mean.tolist(),  # Convert to list for JSON
            'outlier_std': outlier_std.tolist(),    # Convert to list for JSON
            'non_outlier_mean': non_outlier_mean.tolist(),  # Convert to list for JSON
            'non_outlier_std': non_outlier_std.tolist()     # Convert to list for JSON
        }
        
        # Save analysis to CSV
        try:
            # Create a DataFrame with metrics that can be saved to CSV
            analysis_df = pd.DataFrame({
                'Metric': ['total_frames', 'outlier_count', 'outlier_percentage'],
                'Value': [outlier_stats['total_frames'], outlier_stats['outlier_count'], outlier_stats['outlier_percentage']]
            })
            analysis_df.to_csv(os.path.join(output_dir, 'outlier_analysis.csv'), index=False)
            
            # Save outlier indices to a separate file
            with open(os.path.join(output_dir, 'outliers_indices.txt'), 'w') as f:
                f.write(','.join(map(str, sorted(outlier_indices_list))))
        except Exception as e:
            print(f"[WARNING] Error saving outlier analysis: {e}")
        
        return outlier_stats
    
    @staticmethod
    def visualize_outliers(features: np.ndarray, 
                           outlier_indices: Set[int], 
                           output_dir: str = 'outlier_visualization'):
        """
        Visualize outliers using dimensionality reduction
        
        Args:
            features (np.ndarray): Feature matrix
            outlier_indices (Set[int]): Indices of outlier frames
            output_dir (str): Directory to save visualizations
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("[WARNING] Matplotlib/Seaborn not available for visualization")
            return
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert outlier indices to list of Python integers
        outlier_indices_list = [int(i) for i in outlier_indices]
        
        # Check if we have enough data for visualization
        if features.shape[0] < 2 or features.shape[1] < 2:
            print("[WARNING] Not enough data for visualization")
            return
        
        try:
            # Dimensionality reduction with PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(StandardScaler().fit_transform(features))
            
            # Prepare data for plotting
            is_outlier = np.zeros(features.shape[0], dtype=bool)
            for idx in outlier_indices_list:
                if 0 <= idx < len(is_outlier):  # Make sure index is valid
                    is_outlier[idx] = True
            
            # Create scatter plot
            plt.figure(figsize=(12, 8))
            plt.scatter(
                features_2d[~is_outlier, 0], 
                features_2d[~is_outlier, 1], 
                c='blue', 
                label='Normal Frames', 
                alpha=0.7
            )
            plt.scatter(
                features_2d[is_outlier, 0], 
                features_2d[is_outlier, 1], 
                c='red', 
                label='Outlier Frames', 
                marker='x', 
                s=100
            )
            
            plt.title('Outlier Detection Visualization (PCA)')
            plt.xlabel('First Principal Component')
            plt.ylabel('Second Principal Component')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'outliers_pca_visualization.png'))
            plt.close()
        
        except Exception as e:
            print(f"[WARNING] Visualization error: {e}")

# Example usage
def main():
    # Simulated feature matrix (replace with actual features)
    np.random.seed(42)
    features = np.random.randn(100, 10)
    
    # Inject some artificial outliers
    features[0:5] *= 10  # Make first 5 rows extreme
    
    # Detect outliers using ensemble method
    outlier_indices = AdvancedOutlierDetector.detect_outliers_ensemble(features)
    
    # Analyze outliers
    outlier_stats = AdvancedOutlierDetector.analyze_outliers(features, outlier_indices)
    print("Outlier Statistics:", outlier_stats)
    
    # Visualize outliers
    AdvancedOutlierDetector.visualize_outliers(features, outlier_indices)

if __name__ == "__main__":
    main()