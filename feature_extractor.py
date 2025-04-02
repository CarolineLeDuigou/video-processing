import cv2
import numpy as np
import pandas as pd
import os
from typing import List, Tuple, Optional
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

class AdvancedFeatureExtractor:
    """
    Comprehensive feature extraction for video frame analysis
    """
    
    @staticmethod
    def calculate_texture_variance(image: np.ndarray) -> float:
        """
        Calculate texture variance using alternative method
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            float: Variance of the image
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(np.var(gray))
    
    @staticmethod
    def calculate_skewness(image: np.ndarray) -> float:
        """
        Calculate skewness of image intensity
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            float: Skewness of image
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(stats.skew(gray.flatten()))
    
    @staticmethod
    def calculate_kurtosis(image: np.ndarray) -> float:
        """
        Calculate kurtosis of image intensity
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            float: Kurtosis of image
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        return float(stats.kurtosis(gray.flatten()))
    
    @staticmethod
    def calculate_entropy(image: np.ndarray) -> float:
        """
        Calculate image entropy
        
        Args:
            image (np.ndarray): Input image
        
        Returns:
            float: Entropy of the image
        """
        # Convert to grayscale if needed
        if len(image.shape) > 2:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Calculate histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = hist / np.sum(hist)
        
        # Calculate entropy
        entropy = -np.sum(hist * np.log2(hist + np.finfo(float).eps))
        return float(entropy)
    
    @classmethod
    def extract_frame_features(cls, 
                              frames: List[str], 
                              verbose: bool = False,
                              output_dir: str = 'feature_analysis') -> Tuple[np.ndarray, List[str]]:
        """
        Extract comprehensive features from a list of frame paths
        
        Args:
            frames (List[str]): Paths to frame images
            verbose (bool): Print detailed extraction information
            output_dir (str): Directory to save analysis results
        
        Returns:
            Tuple of feature matrix and frame names
        """
        if not frames:
            raise ValueError("No frames provided for feature extraction")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        all_features = []
        frame_names = []
        
        # Save original frames for reference
        frames_preview_dir = os.path.join(output_dir, 'frames_preview')
        os.makedirs(frames_preview_dir, exist_ok=True)
        
        print(f"Extracting features from {len(frames)} frames...")
        for frame_index, frame_path in enumerate(frames):
            try:
                # Read image
                image = cv2.imread(frame_path)
                if image is None:
                    print(f"[WARNING] Could not read image: {frame_path}")
                    continue
                
                # Save a copy of the frame for reference
                frame_basename = os.path.basename(frame_path)
                preview_path = os.path.join(frames_preview_dir, frame_basename)
                cv2.imwrite(preview_path, image)
                
                # Convert to grayscale
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                
                # Color histogram features (BGR channels)
                hist_b = cv2.calcHist([image], [0], None, [16], [0, 256])
                hist_g = cv2.calcHist([image], [1], None, [16], [0, 256])
                hist_r = cv2.calcHist([image], [2], None, [16], [0, 256])
                
                # Normalize histograms
                hist_b = cv2.normalize(hist_b, hist_b).flatten()
                hist_g = cv2.normalize(hist_g, hist_g).flatten()
                hist_r = cv2.normalize(hist_r, hist_r).flatten()
                
                # Edge and gradient features
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges > 0) / (edges.shape[0] * edges.shape[1])
                
                # Save edge detection visualization
                if frame_index % 10 == 0 or len(frames) < 20:  # Save every 10th frame or all if few frames
                    edge_viz_dir = os.path.join(output_dir, 'edge_detection')
                    os.makedirs(edge_viz_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(edge_viz_dir, f'edges_{frame_basename}'), edges)
                
                # Gradient features
                sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
                sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
                gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
                
                # Save gradient visualization
                if frame_index % 10 == 0 or len(frames) < 20:
                    grad_viz_dir = os.path.join(output_dir, 'gradient_viz')
                    os.makedirs(grad_viz_dir, exist_ok=True)
                    
                    # Normalize gradient for visualization
                    grad_norm = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                    cv2.imwrite(os.path.join(grad_viz_dir, f'gradient_{frame_basename}'), grad_norm)
                
                # Texture and statistical features
                frame_features = [
                    *hist_b,  # Blue channel histogram
                    *hist_g,  # Green channel histogram
                    *hist_r,  # Red channel histogram
                    edge_density,  # Edge density
                    np.mean(gradient_magnitude),  # Mean gradient magnitude
                    np.std(gradient_magnitude),  # Std of gradient magnitude
                    cls.calculate_texture_variance(image),  # Texture variance
                    cls.calculate_skewness(image),  # Skewness
                    cls.calculate_kurtosis(image),  # Kurtosis
                    cls.calculate_entropy(image),  # Entropy
                    np.mean(gray),  # Mean intensity
                    np.std(gray),  # Standard deviation
                ]
                
                all_features.append(frame_features)
                frame_names.append(os.path.basename(frame_path))
                
                if verbose and (frame_index % 10 == 0 or frame_index == len(frames) - 1):
                    print(f"[INFO] Extracted features for frame {frame_index+1}/{len(frames)} - {frame_path}")
                    print(f"Feature vector length: {len(frame_features)}")
            
            except Exception as e:
                print(f"[ERROR] Failed to extract features from {frame_path}: {e}")
        
        # Convert to numpy array
        features_array = np.array(all_features)
        
        # Validate feature extraction
        if features_array.size == 0:
            raise ValueError("No features could be extracted from the frames")
        
        # Normalize features
        scaler = StandardScaler()
        features_normalized = scaler.fit_transform(features_array)
        
        # Save feature analysis (always, not just in verbose mode)
        cls.save_feature_analysis(features_normalized, frame_names, output_dir)
        
        return features_normalized, frame_names
    
    @staticmethod
    def save_feature_analysis(features: np.ndarray, 
                             frame_names: List[str], 
                             output_dir: str = 'feature_analysis'):
        """
        Save detailed feature analysis
        
        Args:
            features (np.ndarray): Normalized feature matrix
            frame_names (List[str]): Corresponding frame names
            output_dir (str): Directory to save analysis results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Feature statistics
        feature_stats = pd.DataFrame({
            'Mean': np.mean(features, axis=0),
            'Std': np.std(features, axis=0),
            'Min': np.min(features, axis=0),
            'Max': np.max(features, axis=0)
        })
        
        # Save feature statistics
        feature_stats.to_csv(os.path.join(output_dir, 'feature_statistics.csv'), index=False)
        
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(viz_dir, exist_ok=True)
        
        # Try to create visualization if matplotlib is available
        try:
            # 1. Features heatmap
            plt.figure(figsize=(15, 10))
            sns.heatmap(features, cmap='viridis', center=0)
            plt.title('Normalized Frame Features Heatmap')
            plt.xlabel('Feature Index')
            plt.ylabel('Frame')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'features_heatmap.png'))
            plt.close()
            
            # 2. Feature correlation matrix
            plt.figure(figsize=(12, 10))
            corr_matrix = np.corrcoef(features.T)
            sns.heatmap(corr_matrix, cmap='coolwarm', center=0, 
                        square=True, linewidths=.5, annot=False)
            plt.title('Feature Correlation Matrix')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'feature_correlation.png'))
            plt.close()
            
            # 3. Feature distributions
            n_features = min(10, features.shape[1])  # Show up to 10 features
            plt.figure(figsize=(15, n_features * 2))
            for i in range(n_features):
                plt.subplot(n_features, 1, i+1)
                sns.histplot(features[:, i], kde=True)
                plt.title(f'Feature {i+1} Distribution')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'feature_distributions.png'))
            plt.close()
            
            # 4. Feature variation over frames
            plt.figure(figsize=(15, 8))
            for i in range(min(5, features.shape[1])):  # Plot first 5 features
                plt.plot(features[:, i], label=f'Feature {i+1}')
            plt.title('Feature Variation Across Frames')
            plt.xlabel('Frame Index')
            plt.ylabel('Normalized Feature Value')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(viz_dir, 'feature_variation.png'))
            plt.close()
            
            # Save summary text file
            with open(os.path.join(output_dir, 'analysis_summary.txt'), 'w') as f:
                f.write(f"Feature Analysis Summary\n")
                f.write(f"======================\n\n")
                f.write(f"Total frames analyzed: {features.shape[0]}\n")
                f.write(f"Features extracted per frame: {features.shape[1]}\n\n")
                f.write(f"Global statistics have been saved to 'feature_statistics.csv'\n")
                f.write(f"Visualizations saved to the 'visualizations' directory\n")
        
        except ImportError:
            print("[WARNING] Matplotlib/Seaborn not available for visualization")
        except Exception as e:
            print(f"[WARNING] Visualization error: {e}")

# Example usage
def main():
    # Example frame paths (replace with your actual paths)
    frame_paths = [
        'video_output/frames/frame_0000.jpg',
        'video_output/frames/frame_0001.jpg',
        # Add more frame paths
    ]
    
    try:
        features, names = AdvancedFeatureExtractor.extract_frame_features(
            frame_paths, 
            verbose=True,
            output_dir='feature_analysis_results'
        )
        print(f"Extracted features shape: {features.shape}")
    except Exception as e:
        print(f"Feature extraction failed: {e}")

if __name__ == "__main__":
    main()