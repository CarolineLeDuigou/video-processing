import os
import numpy as np
import networkx as nx
import json
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional, Dict, Any

# Machine Learning and Image Processing Libraries
import cv2
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

class AdvancedFrameReorderer:
    """
    Comprehensive frame reordering system with multiple sophisticated strategies
    
    Supports various reordering techniques:
    1. Optical Flow Continuity
    2. Feature Similarity Graph
    3. Topological Sorting
    4. Temporal Consistency Analysis
    5. Trajectory-based Reordering
    """
    
    @staticmethod
    def calculate_optical_flow(frame1_path: str, 
                                frame2_path: str, 
                                method: str = 'farneback') -> Optional[Dict[str, float]]:
        """
        Calculate advanced optical flow metrics between two frames
        
        Args:
            frame1_path: Path to first frame
            frame2_path: Path to second frame
            method: Optical flow method ('farneback' or 'lucas_kanade')
            
        Returns:
            Dictionary of flow metrics or None if calculation failed
        """
        # Read frames
        frame1 = cv2.imread(frame1_path)
        frame2 = cv2.imread(frame2_path)
        
        if frame1 is None or frame2 is None:
            print(f"[ERROR] Failed to read frames: {frame1_path} or {frame2_path}")
            return None
        
        # Convert to grayscale
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
        
        try:
            # Calculate optical flow
            if method == 'farneback':
                # Farneback dense optical flow
                flow = cv2.calcOpticalFlowFarneback(
                    gray1, gray2, None, 
                    pyr_scale=0.5, levels=5, winsize=15, 
                    iterations=3, poly_n=5, poly_sigma=1.2, 
                    flags=0
                )
                
                # Calculate magnitude and angle
                mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                
                # Calculate flow metrics
                avg_magnitude = float(np.mean(mag))
                max_magnitude = float(np.max(mag))
                std_magnitude = float(np.std(mag))
                
                # Calculate flow direction consistency
                flow_x = flow[..., 0]
                flow_y = flow[..., 1]
                
                # Positive flow ratio (percentage of vectors with positive x or y component)
                pos_x_ratio = float(np.sum(flow_x > 0) / flow_x.size)
                pos_y_ratio = float(np.sum(flow_y > 0) / flow_y.size)
                
                # Flow consistency metrics
                flow_direction = np.arctan2(flow_y, flow_x)
                flow_direction_std = float(np.std(flow_direction))
                
            elif method == 'lucas_kanade':
                # Lucas-Kanade sparse optical flow
                # Find good features to track
                corners = cv2.goodFeaturesToTrack(gray1, maxCorners=100, qualityLevel=0.01, minDistance=10)
                
                if corners is None or len(corners) < 5:
                    return None  # Not enough good features
                
                # Calculate optical flow
                new_corners, status, _ = cv2.calcOpticalFlowPyrLK(gray1, gray2, corners, None)
                
                # Filter valid points
                good_corners = corners[status == 1]
                good_new_corners = new_corners[status == 1]
                
                if len(good_corners) < 5:
                    return None  # Not enough tracked points
                
                # Calculate flow vectors
                flow_vectors = good_new_corners - good_corners
                
                # Calculate flow metrics
                magnitudes = np.sqrt(flow_vectors[:, 0, 0]**2 + flow_vectors[:, 0, 1]**2)
                avg_magnitude = float(np.mean(magnitudes))
                max_magnitude = float(np.max(magnitudes))
                std_magnitude = float(np.std(magnitudes))
                
                # Direction calculations
                flow_directions = np.arctan2(flow_vectors[:, 0, 1], flow_vectors[:, 0, 0])
                flow_direction_std = float(np.std(flow_directions))
                
                # Positive flow ratio
                pos_x_ratio = float(np.sum(flow_vectors[:, 0, 0] > 0) / len(flow_vectors))
                pos_y_ratio = float(np.sum(flow_vectors[:, 0, 1] > 0) / len(flow_vectors))
            else:
                raise ValueError(f"Unknown optical flow method: {method}")
            
            # Return flow metrics dictionary
            return {
                'avg_magnitude': avg_magnitude,
                'max_magnitude': max_magnitude,
                'std_magnitude': std_magnitude,
                'pos_x_ratio': pos_x_ratio,
                'pos_y_ratio': pos_y_ratio,
                'direction_std': flow_direction_std,
                'method': method
            }
        except Exception as e:
            print(f"[ERROR] Optical flow calculation failed: {e}")
            return None
    
    @staticmethod
    def build_similarity_graph(frames: List[str], 
                            features: np.ndarray, 
                            similarity_threshold: float = 0.7) -> nx.DiGraph:
        """
        Build a similarity graph between frames

        Args:
            frames: List of frame paths
            features: Feature matrix (n_frames x n_features)
            similarity_threshold: Minimum similarity to create an edge

        Returns:
            Directed graph representing frame similarities
        """
        if len(frames) != features.shape[0]:
            raise ValueError("Number of frames and features rows must match")

        G = nx.DiGraph()

        # Add nodes
        for i, frame_path in enumerate(frames):
            G.add_node(i, path=frame_path)

        # Normalize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)

        similarity_matrix = cosine_similarity(features_scaled)

        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                sim = similarity_matrix[i, j]

                if sim >= similarity_threshold:
                    # 🔁 Inversé ici : arête de j → i
                    G.add_edge(j, i, weight=sim, similarity=sim)

                    if sim >= 0.9:
                        G.add_edge(i, j, weight=sim * 0.9, similarity=sim * 0.9)

        # Assurer la connexité
        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            for i, comp1 in enumerate(components[:-1]):
                for comp2 in components[i + 1:]:
                    best_sim = -1
                    best_pair = None
                    for node1 in comp1:
                        for node2 in comp2:
                            sim = similarity_matrix[node1, node2]
                            if sim > best_sim:
                                best_sim = sim
                                best_pair = (node2, node1)  # 🔁 Note encore ici : j → i

                    if best_pair is not None:
                        G.add_edge(best_pair[0], best_pair[1], weight=best_sim, similarity=best_sim)
                        G.add_edge(best_pair[1], best_pair[0], weight=best_sim * 0.8, similarity=best_sim * 0.8)

        return G
    
    @classmethod
    def reorder_by_topological_sort(cls, 
                                    frames: List[str], 
                                    features: np.ndarray, 
                                    similarity_threshold: float = 0.7) -> List[str]:
        """
        Reorder frames using topological sorting of similarity graph
        
        Args:
            frames: List of frame paths
            features: Feature matrix
            similarity_threshold: Minimum similarity threshold
            
        Returns:
            List of reordered frame paths
        """
        # Build similarity graph
        G = cls.build_similarity_graph(frames, features, similarity_threshold)
        
        # Add missing edges to ensure DAG
        for i in range(len(frames)):
            for j in range(i + 1, len(frames)):
                if not G.has_edge(i, j) and not G.has_edge(j, i):
                    # Add weak edge based on index proximity
                    proximity_weight = 1.0 / (abs(j - i) + 1)
                    G.add_edge(i, j, weight=proximity_weight, similarity=proximity_weight)

        # Remove cycles to ensure DAG
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G)
                min_weight = float('inf')
                min_edge = None

                for edge_data in cycle:
                    if len(edge_data) == 3:
                        u, v, data = edge_data
                    elif len(edge_data) == 2:
                        u, v = edge_data
                        data = G.get_edge_data(u, v)
                    else:
                        continue

                    if data['weight'] < min_weight:
                        min_weight = data['weight']
                        min_edge = (u, v)

                if min_edge:
                    G.remove_edge(*min_edge)
            except nx.NetworkXNoCycle:
                break

        # Perform topological sort
        try:
            sorted_nodes = list(nx.topological_sort(G))

            # Heuristique : détecter si l'ordre est inversé selon les indices dans les noms de fichiers
            def extract_index(filename):
                name = os.path.splitext(os.path.basename(filename))[0]
                digits = ''.join(filter(str.isdigit, name))
                return int(digits) if digits else -1

            first_idx = extract_index(frames[sorted_nodes[0]])
            last_idx = extract_index(frames[sorted_nodes[-1]])

            if first_idx > last_idx:
                print("[INFO] Topological order seems inverted, correcting...")
                sorted_nodes.reverse()

            reordered_frames = [frames[i] for i in sorted_nodes]
            return reordered_frames

        except nx.NetworkXUnfeasible:
            print("[WARNING] Graph is not a DAG, using original order")
            return frames
    
    @classmethod
    def reorder_by_optical_flow_continuity(cls, 
                                           frames: List[str], 
                                           max_window: int = 5) -> List[str]:
        """
        Reorder frames based on optical flow continuity
        
        Args:
            frames: List of frame paths
            max_window: Maximum window size for optical flow calculation
            
        Returns:
            List of reordered frame paths
        """
        if len(frames) <= 2:
            return frames  # Nothing to reorder
        
        # Create a complete directed graph where edges represent flow "goodness"
        G = nx.DiGraph()
        
        # Add nodes
        for i, frame_path in enumerate(frames):
            G.add_node(i, path=frame_path)
        
        # Calculate optical flow between pairs of frames within the window
        for i in range(len(frames)):
            # Look ahead within the window
            for j in range(i + 1, min(i + max_window + 1, len(frames))):
                # Calculate optical flow
                flow_metrics = cls.calculate_optical_flow(frames[i], frames[j])
                
                if flow_metrics is None:
                    continue  # Skip if flow calculation failed
                
                # Edge weight represents how good the flow is (higher is better)
                # We use a combination of metrics to determine "goodness"
                
                # Favor moderate, consistent flow
                flow_magnitude = flow_metrics['avg_magnitude']
                flow_consistency = 1.0 / (flow_metrics['direction_std'] + 0.1)
                
                # Penalize extreme magnitudes (too small or too large)
                magnitude_penalty = abs(flow_magnitude - 5.0) / 10.0
                
                # Higher weight means better flow
                flow_goodness = flow_consistency - magnitude_penalty
                
                # Add edge with the goodness as weight
                G.add_edge(i, j, weight=flow_goodness, flow=flow_metrics)
        
        # Find optimal path (longest path with highest weights)
        # For this, we use a modified Dijkstra algorithm with negative weights
        
        # Convert to negative weights for longest path
        for u, v, data in G.edges(data=True):
            data['weight'] = -data['weight']
        
        # Add a super source and target
        G.add_node('source')
        G.add_node('target')
        
        # Connect source to all nodes and all nodes to target
        for i in range(len(frames)):
            G.add_edge('source', i, weight=0)
            G.add_edge(i, 'target', weight=0)
        
        # Find shortest path with negative weights
        try:
            path = nx.shortest_path(G, 'source', 'target', weight='weight')
            
            # Remove source and target
            path = path[1:-1]
            
            # Get frame paths in the optimal order
            reordered_frames = [frames[i] for i in path]
            
            # Add any skipped frames at the end
            included = set(path)
            skipped = [i for i in range(len(frames)) if i not in included]
            
            for i in skipped:
                reordered_frames.append(frames[i])
            
            return reordered_frames
        
        except (nx.NetworkXNoPath, nx.NetworkXError):
            print("[WARNING] No valid path found, using original order")
            return frames
    
    @classmethod
    def advanced_frame_reordering(cls, 
                              frames: List[str], 
                              features: np.ndarray, 
                              method: str = 'hybrid',
                              output_dir: str = 'reordering_analysis') -> List[str]:
        """
        Advanced frame reordering with multiple strategies
        
        Args:
            frames (List[str]): List of frame paths
            features (np.ndarray): Frame feature matrix
            method (str): Reordering method
            output_dir (str): Output directory for analysis
        
        Returns:
            Reordered frame list
        """
        if method == 'topological':
            reordered_frames = cls.reorder_by_topological_sort(frames, features)
        elif method == 'optical_flow':
            reordered_frames = cls.reorder_by_optical_flow_continuity(frames)
        elif method == 'feature_matching':
            # Implémentation simple de correspondance de caractéristiques
            # Utiliser la similarité cosinus pour réordonner
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            similarity_matrix = cosine_similarity(features_scaled)
            
            # Ordonner les frames par similarité avec la première frame
            sim_to_first = similarity_matrix[0, :]
            order = np.argsort(-sim_to_first)  # Tri décroissant (plus similaire en premier)
            reordered_frames = [frames[i] for i in order]
        else:  # hybrid method
            # Try topological sort first
            try:
                reordered_frames = cls.reorder_by_topological_sort(frames, features)
            except Exception as e:
                print(f"[WARNING] Topological sort failed: {e}")
                # Fallback to optical flow continuity
                reordered_frames = cls.reorder_by_optical_flow_continuity(frames)
        
        # Create visualization before returning result
        os.makedirs(output_dir, exist_ok=True)
        original_frames_paths = frames.copy()
        reordered_frames_paths = reordered_frames
            
        # Analyze and save reordering visualization
        cls.analyze_frame_ordering(original_frames_paths, reordered_frames_paths, output_dir)   
            
        return reordered_frames
    

    @staticmethod
    def reorder_by_track_id(track_history: Dict[int, Dict[str, Any]], track_id: int) -> List[int]:
        """
        Reorder frames based on the trajectory of a specific track_id.
        Determines if the person is approaching or moving away from the camera
        based on bbox size trends, and orders frames accordingly.
        
        Args:
            track_history: Dictionary containing frame indices and bounding boxes per track_id
            track_id: The specific object ID to track
            
        Returns:
            List of frame indices sorted to create a coherent movement
        """
        if not track_history:
            print("[WARNING] Track history is empty.")
            return []
            
        if track_id not in track_history:
            print(f"[WARNING] Track ID {track_id} not found in track history.")
            return []
        
        frames = track_history[track_id].get("frames", [])
        bboxes = track_history[track_id].get("bboxes", [])
        
        if not frames or not bboxes or len(frames) != len(bboxes):
            print(f"[WARNING] Invalid track data for ID {track_id}.")
            return []

        # Compute bbox areas for each frame
        frame_areas = []
        for frame_idx, bbox in zip(frames, bboxes):
            try:
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                frame_areas.append((frame_idx, area))
            except (IndexError, TypeError):
                print(f"[WARNING] Invalid bbox data for frame {frame_idx}.")
                continue

        if not frame_areas:
            return []
        
        # Detect if the primary movement is approaching or moving away
        # We'll use the 10 largest and 10 smallest bounding boxes to determine
        sorted_by_area = sorted(frame_areas, key=lambda x: x[1])
        
        # Calculer la corrélation entre la taille et l'index temporel d'origine
        frame_indices = [frames.index(idx) for idx, _ in frame_areas]
        areas = [area for _, area in frame_areas]

        # Calculer le coefficient de corrélation de Pearson
        import numpy as np
        correlation = np.corrcoef(frame_indices, areas)[0, 1]

        # Si la corrélation est positive, les grandes bounding boxes apparaissent 
        # plus tard dans la séquence originale, ce qui suggère que la personne s'approche
        prefer_approaching = correlation > 0

        print(f"[INFO] Track ID {track_id}: Corrélation taille-index: {correlation:.2f} - {'approaching' if prefer_approaching else 'moving away'}")
        
        # Sort frames by area: small to large (approaching) or large to small (moving away)
        if prefer_approaching:
            # Person approaching camera - sort from small to large
            sorted_frames = [idx for idx, _ in sorted(frame_areas, key=lambda x: x[1])]
        else:
            # Person moving away from camera - sort from large to small
            sorted_frames = [idx for idx, _ in sorted(frame_areas, key=lambda x: x[1], reverse=True)]
        
        return sorted_frames


    @staticmethod
    def analyze_frame_ordering(original_frames: List[str], 
                                reordered_frames: List[str], 
                                output_dir: str = 'analysis') -> Dict[str, Any]:
        """
        Analyze frame reordering
        
        Args:
            original_frames: Original frame paths
            reordered_frames: Reordered frame paths
            output_dir: Output directory for analysis
            
        Returns:
            Dictionary with analysis metrics
        """
        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract base filenames
        original_names = [os.path.basename(f) for f in original_frames]
        reordered_names = [os.path.basename(f) for f in reordered_frames]
        
        # Compute frame changes
        frames_changed = 0
        comparison_length = min(len(original_names), len(reordered_names))
        
        for i in range(comparison_length):
            if original_names[i] != reordered_names[i]:
                frames_changed += 1
        
        # Metrics
        analysis = {
            'total_original_frames': len(original_frames),
            'total_reordered_frames': len(reordered_frames),
            'frames_changed_position': frames_changed,
            'order_change_percentage': (frames_changed / comparison_length) * 100 if comparison_length > 0 else 0
        }
        
        # Prepare frame positions mapping
        original_to_reordered = {}
        for i, name in enumerate(original_names):
            try:
                new_pos = reordered_names.index(name)
                original_to_reordered[i] = new_pos
            except ValueError:
                # Frame not found in reordered list
                pass
        
        # Visualization
        try:
            # Plot frame order changes
            plt.figure(figsize=(12, 8))
            
            # Points representing frame position changes
            positions = list(original_to_reordered.items())
            x = [p[0] for p in positions]
            y = [p[1] for p in positions]
            
            plt.scatter(x, y, alpha=0.7)
            plt.plot(x, y, 'r-', alpha=0.3)
            
            # Draw identity line for reference
            plt.plot([0, max(comparison_length-1, 1)], [0, max(comparison_length-1, 1)], 
                   'g--', alpha=0.5, label='Original Order')
            
            plt.xlabel('Original Position')
            plt.ylabel('New Position')
            plt.title('Frame Reordering Map')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'frame_position_changes.png'))
            plt.close()
            
            # Histogram of position changes
            position_shifts = [abs(y - x) for x, y in positions]
            
            plt.figure(figsize=(10, 6))
            plt.hist(position_shifts, bins=20)
            plt.xlabel('Position Shift Magnitude')
            plt.ylabel('Count')
            plt.title('Distribution of Frame Position Shifts')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'position_shift_histogram.png'))
            plt.close()
            
            # Original vs Reordered frame sequence visualization
            plt.figure(figsize=(12, 6))
            plt.subplot(121)
            plt.title('Original Frame Order')
            plt.plot(range(len(original_names)), range(len(original_names)), 'b-')
            plt.xlabel('Sequence Position')
            plt.ylabel('Frame Position')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(122)
            plt.title('Reordered Frame Order')
            plt.scatter(range(len(positions)), y, alpha=0.7)
            plt.xlabel('Sequence Position')
            plt.ylabel('Original Frame Position')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'frame_order_comparison.png'))
            plt.close()
        
        except Exception as e:
            print(f"Visualization error: {e}")
        
        # Save analysis
        with open(os.path.join(output_dir, 'frame_ordering_analysis.json'), 'w') as f:
            json.dump(analysis, f, indent=2)
        
        # Create text report
        report_path = os.path.join(output_dir, 'frame_ordering_report.txt')
        with open(report_path, 'w') as f:
            f.write("Frame Reordering Analysis\n")
            f.write("=======================\n\n")
            f.write(f"Total original frames: {len(original_frames)}\n")
            f.write(f"Total reordered frames: {len(reordered_frames)}\n\n")
            f.write(f"Frames that changed position: {frames_changed} ")
            f.write(f"({analysis['order_change_percentage']:.1f}%)\n\n")
            
            if frames_changed > 0:
                f.write("Distribution of position shifts:\n")
                f.write(f"  Maximum shift: {max(position_shifts) if position_shifts else 0} positions\n")
                f.write(f"  Average shift: {sum(position_shifts)/len(position_shifts) if position_shifts else 0:.1f} positions\n")
                
                # List largest shifts
                if position_shifts:
                    largest_shifts = sorted([(i, j, j-i) for i, j in positions], 
                                         key=lambda x: abs(x[2]), reverse=True)[:5]
                    
                    f.write("\nLargest position shifts:\n")
                    for orig, new, shift in largest_shifts:
                        f.write(f"  Frame at position {orig} moved to {new} (shift of {shift} positions)\n")
        
        return analysis
    
    @staticmethod
    def detect_frame_sequence_direction(track_data):
        """
        Détecter intelligemment si la séquence est dans le bon ordre
        """
        # Analyser la trajectoire et la taille des bounding boxes
        if 'centers' not in track_data or 'bboxes' not in track_data:
            return 'unknown'
        
        centers = track_data['centers']
        bbox_areas = [
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) 
            for bbox in track_data['bboxes']
        ]
        
        # Vérifier s'il y a suffisamment de données
        if len(centers) < 5 or len(bbox_areas) < 5:
            return 'forward'  # Valeur par défaut
        
        # Calculer la progression spatiale et temporelle
        try:
            spatial_progression = np.corrcoef(range(len(centers)), 
                                            [c[1] for c in centers])[0, 1]  # Utiliser la coordonnée Y
            area_progression = np.corrcoef(range(len(bbox_areas)), bbox_areas)[0, 1]
            
            # Décider de l'orientation
            if spatial_progression < -0.5 or area_progression < -0.5:
                return 'reversed'
            return 'forward'
        except Exception:
            return 'forward'  # Valeur par défaut en cas d'erreur

    @classmethod
    def calculate_frame_order_coherence(cls, frames, reordered_frames):
        """
        Calculer un score de cohérence de réordonnancement
        """
        def calculate_frame_similarity(frame1, frame2):
            """Calculer la similarité entre deux frames"""
            # Convertir en niveaux de gris pour simplifier
            gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
            
            # Utiliser la différence structurelle (SSIM)
            from skimage.metrics import structural_similarity as ssim
            return ssim(gray1, gray2)
        
        # Calculer les similarités entre frames adjacentes
        def calculate_adjacency_similarities(frames_list):
            return [
                calculate_frame_similarity(frames_list[i], frames_list[i+1]) 
                for i in range(len(frames_list)-1)
            ]
        
        # Comparer les distributions de similarités
        original_similarities = calculate_adjacency_similarities(frames)
        reordered_similarities = calculate_adjacency_similarities(reordered_frames)
        
        # Calculer des métriques statistiques
        import numpy as np
        original_mean = np.mean(original_similarities)
        reordered_mean = np.mean(reordered_similarities)
        
        # Score de cohérence basé sur la similarité moyenne
        coherence_score = 1 - abs(original_mean - reordered_mean)
        
        return coherence_score
    
    @staticmethod
    def detect_track_direction(track_data):
        """
        Détecter intelligemment la direction du mouvement
        """
        if not track_data or 'bboxes' not in track_data or 'centers' not in track_data:
            return 'unknown'

        bboxes = track_data['bboxes']
        centers = track_data['centers']
        
        # Vérification de la validité des données
        if len(bboxes) < 5 or len(centers) < 5:
            return 'forward'
        
        # Calculer la progression des tailles
        areas = [
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) 
            for bbox in bboxes
        ]
        
        # Calculer la progression verticale
        vertical_progression = [center[1] for center in centers]
        
        # Calculs statistiques
        start_area = areas[0]
        end_area = areas[-1]
        area_ratio = end_area / start_area if start_area > 0 else 1
        
        # Progression verticale (montée/descente)
        start_y = vertical_progression[0]
        end_y = vertical_progression[-1]
        vertical_movement = end_y - start_y
        
        # Critères de décision
        is_approaching = (
            area_ratio > 1.2 or  # Augmentation significative de la taille
            vertical_movement > 0  # Descente dans l'image
        )
        
        return 'approaching' if is_approaching else 'moving_away'

# Example usage
def main():
    # Simulated frame paths and features
    np.random.seed(42)
    frames = [f'frame_{i}.jpg' for i in range(100)]
    features = np.random.rand(100, 50)  # Random feature matrix
    
    # Reorder frames
    reordered_frames = AdvancedFrameReorderer.advanced_frame_reordering(
        frames, 
        features, 
        method='hybrid'
    )
    
    # Analyze reordering
    analysis = AdvancedFrameReorderer.analyze_frame_ordering(
        frames, 
        reordered_frames
    )
    
    print("Reordering Analysis:", analysis)

if __name__ == "__main__":
    main()