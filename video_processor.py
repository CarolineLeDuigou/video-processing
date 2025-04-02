import os
import cv2
import json
import logging
import shutil
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
from datetime import datetime
# import sys
# sys.path.append('Yolov5_StrongSORT_OSNet')
import sys
# sys.path.append("video_processing/Yolov5_StrongSORT_OSNet")
# sys.path.append("Yolov5_StrongSORT_OSNet")

# Imports des modules personnalisés
from configs import VideoProcessingConfig
from utils import VideoUtils
from feature_extractor import AdvancedFeatureExtractor
from outlier_detector import AdvancedOutlierDetector
from frame_reorderer import AdvancedFrameReorderer
from trackers.person_tracker import PersonTracker
from AutoTrackReorderUtils import AutoTrackReorderUtils
import traceback
import matplotlib.pyplot as plt
import seaborn as sns


# Essayer d'initialiser tqdm pour les barres de progression
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False
    print("[INFO] tqdm non disponible, les barres de progression seront désactivées")

# Essayez d'importer les modules spécifiques avec gestion d'erreur
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("[AVERTISSEMENT] Ultralytics (YOLO) non disponible. L'analyse d'objets sera limitée.")
    YOLO_AVAILABLE = False

try:
    from sklearn.decomposition import PCA
    from sklearn import preprocessing
    SKLEARN_AVAILABLE = True
except ImportError:
    print("[AVERTISSEMENT] Scikit-learn non disponible. La détection d'anomalies par PCA sera désactivée.")
    SKLEARN_AVAILABLE = False

class VideoProcessor:
    """Classe pour traiter et corriger des vidéos corrompues."""
    
    def __init__(self, config=None):
        """Initialise le processeur vidéo avec la configuration fournie."""
        # Configuration par défaut
        self.default_config = {
            'max_frames': None,                # Traiter toutes les frames
            'output_root': 'video_output',     # Répertoire racine pour les sorties
            'outlier_detection_methods': ['zscore', 'pca'],  # Méthodes de détection d'outliers
            'reordering_method': 'fused',   # Méthode de réordonnancement
            'verbose_feature_extraction': False,      # Afficher détails d'extraction
            'visualize_outliers': False,       # Visualiser les outliers
            'tracking_confidence': 0.8,        # Seuil de confiance pour le tracking
            'skip_corrupted_segments': True,   # Ignorer segments corrompus
            'min_frames_for_processing': 5,    # Nb minimum de frames pour traitement
            'fallback_reordering': 'feature_matching', # Méthode de secours
            'yolo_model_path': 'yolov8m.pt',   # Chemin du modèle YOLO
            'save_all_visualizations': True,   # Sauvegarder toutes les visualisations
            'create_processing_report': True,  # Créer rapport de traitement
            'analysis_subdirs': True, 
            'analyze_objects': True,
            'object_analysis': {
                'export_csv': True,
                'save_all_frames': True,
                'visualize_detections': True
            }, 
            'tracking_config': {
                'color_similarity_threshold': 0.6,
                'max_frames_between_match': 100,
                'max_iou_distance': 0.1, 
                'max_tracks': 2  # Maximum 2 tracks
            },                                 
            'target_track_id': None,
            'tracker': 'botsort',
            'output_fps': 30.0
        }

        # Fusionner la configuration par défaut avec celle fournie
        self.config = self.default_config.copy()
        if config:
            self.config.update(config)
        
        # Configurer le logger
        self.setup_logger()
        
        # Initialiser le modèle YOLO si nécessaire
        self.yolo_model = None
        if self.config['reordering_method'] in ['tracking', 'fused'] and YOLO_AVAILABLE:
            try:
                if os.path.exists(self.config['yolo_model_path']):
                    self.yolo_model = YOLO(self.config['yolo_model_path'])
                    self.logger.info(f"Modèle YOLO chargé: {self.config['yolo_model_path']}")
                else:
                    self.logger.warning("Modèle YOLO non trouvé.")
                    self.config['reordering_method'] = self.config['fallback_reordering']
            except Exception as e:
                self.logger.error(f"Erreur chargement YOLO: {e}")
                self.config['reordering_method'] = self.config['fallback_reordering']
        
      # Initialiser le tracker de personne en fonction du type choisi
        try:
            if self.config.get('tracker') == 'botsort':
                # Utiliser BoT-SORT via supervision
                from trackers.ultralytics_bot_tracker import UltralyticsTracker
                tracker_config = {
                    'yolo_model_path': self.config['yolo_model_path'],
                    'confidence_threshold': self.config.get('tracking_confidence', 0.6),
                    'tracker_type': "botsort",
                    'visualization_enabled': self.config.get('save_all_visualizations', True)
                }
                
                # Remove the max_tracks parameter from initialization
                self.person_tracker = UltralyticsTracker(**{k: v for k, v in tracker_config.items() if k != 'max_tracks'})
                self.logger.info("BoT-SORT via Ultralytics activé")
            
            elif self.config.get('tracker') == 'bytetrack':
                from trackers.ultralytics_bot_tracker import UltralyticsTracker
                self.person_tracker = UltralyticsTracker(
                    yolo_model_path=self.config['yolo_model_path'],
                    confidence_threshold=self.config.get('tracking_confidence', 0.6),
                    tracker_type="bytetrack",
                    visualization_enabled=self.config.get('save_all_visualizations', True)
                )
                self.logger.info("ByteTrack via Ultralytics activé")

            elif self.config.get('tracker') == 'strongsort':
                # Utiliser StrongSORT (code existant)
                try:
                    from trackers.person_tracker_strongsort import StrongPersonTracker
                    self.person_tracker = StrongPersonTracker(
                        yolo_model_path=self.config['yolo_model_path']
                    )
                    self.logger.info("StrongSORT activé")
                except Exception as e:
                    self.logger.warning(f"Impossible d'initialiser StrongSORT: {e}. Utilisation de DeepSORT comme fallback.")
                    # Fallback vers DeepSORT
                    from trackers.person_tracker import PersonTracker
                    self.person_tracker = PersonTracker(
                        yolo_model_path=self.config['yolo_model_path'],
                        confidence_threshold=self.config.get('tracking_confidence', 0.6),
                        color_similarity_threshold=self.config.get('tracking_config', {}).get('color_similarity_threshold', 0.7),
                        max_frames_between_match=self.config.get('tracking_config', {}).get('max_frames_between_match', 5)
                    )
                    self.logger.info("DeepSORT activé (fallback)")
            else:
                # Utiliser DeepSORT par défaut
                from trackers.person_tracker import PersonTracker
                self.person_tracker = PersonTracker(
                    yolo_model_path=self.config['yolo_model_path'],
                    confidence_threshold=self.config.get('tracking_confidence', 0.6),
                    color_similarity_threshold=self.config.get('tracking_config', {}).get('color_similarity_threshold', 0.7),
                    max_frames_between_match=self.config.get('tracking_config', {}).get('max_frames_between_match', 5)
                )
                self.logger.info("DeepSORT activé")
            
            self.logger.info("Tracker de personnes initialisé")
        except Exception as e:
            self.logger.warning(f"Impossible d'initialiser le tracker de personnes: {e}")
            self.person_tracker = None
    
    def setup_logger(self):
        """Configure le logger pour le processeur vidéo."""
        # Créer le répertoire des logs
        log_dir = os.path.join(self.config['output_root'], 'logs')
        os.makedirs(log_dir, exist_ok=True)
        
        # Configurer le logger
        self.logger = logging.getLogger('VideoProcessor')
        self.logger.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        
        # Handler pour la console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        
        # Handler pour le fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(os.path.join(log_dir, f'video_processor_{timestamp}.log'))
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(formatter)
        
        # Ajouter les handlers
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)
        
        self.logger.info("VideoProcessor initialisé")
        self.logger.info(f"Configuration: {self.config}")

    def reorder_by_tracking(self, person_tracks: Dict[int, Dict[str, Any]]) -> List[np.ndarray]:
        self.logger.info("Réordonnancement par tracking")
        if not person_tracks:
            self.logger.warning("Aucune track détectée")
            return []
        person_tracks = AutoTrackReorderUtils.auto_reverse_all_tracks(person_tracks)
        dominant_track = max(person_tracks.items(), key=lambda x: len(x[1]['frames']))[1]
        frame_indices = dominant_track['frames']
        return [cv2.imread(f"video_output/frames/frame_{i:04d}.jpg") for i in frame_indices]

    def reorder_fused(self, frames: List[np.ndarray], features: np.ndarray, 
                 outlier_indices: Set[int] = None, 
                 person_tracks: Optional[Dict[int, Dict[str, Any]]] = None,
                 output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Réordonne les frames en combinant tracking et caractéristiques visuelles
        
        Args:
            frames: Liste des frames à réordonner
            features: Caractéristiques des frames
            outlier_indices: Indices des frames aberrantes (outliers)
            person_tracks: Tracks de personnes (si déjà calculées)
            output_dir: Répertoire de sortie pour visualisations
            
        Returns:
            Liste de frames réordonnées
        """
        self.logger.info("Réordonnancement hybride (tracking + features)")
        
        # Filtrer les outliers si présents
        if outlier_indices:
            valid_indices = list(set(range(len(frames))) - outlier_indices)
            valid_frames = [frames[i] for i in valid_indices]
            valid_features = features[valid_indices]
        else:
            valid_indices = list(range(len(frames)))
            valid_frames = frames
            valid_features = features
        
        # Obtenir les tracks si non fournies
        if not person_tracks and self.person_tracker:
            tracking_results = self.person_tracker.detect_and_track_person(valid_frames, output_dir=output_dir)
            person_tracks = tracking_results.get('person_tracks', {})
        
        # Si aucune track disponible, utiliser l'ordre d'origine
        if not person_tracks:
            self.logger.warning("Aucune track disponible pour le réordonnancement hybride")
            return valid_frames
        
        # Corriger l'orientation des tracks
        person_tracks = AutoTrackReorderUtils.auto_reverse_all_tracks(person_tracks)
        
        # Trouver la track dominante (la plus longue)
        best_track_id = max(person_tracks.keys(), key=lambda k: len(person_tracks[k]['frames']))
        dominant_track = person_tracks[best_track_id]
        
        # Obtenir les indices des frames de la track dominante
        track_frames = dominant_track['frames']
        self.logger.info(f"Utilisation de la track {best_track_id} avec {len(track_frames)} frames")
        
        # Vérifier que les indices sont valides
        valid_track_frames = [idx for idx in track_frames if 0 <= idx < len(valid_frames)]
        
        if len(valid_track_frames) < 3:
            self.logger.warning("Trop peu de frames valides dans la track dominante")
            return valid_frames
        
        # Extraire les features des frames de la track
        track_features = valid_features[[valid_indices.index(idx) for idx in valid_track_frames]]
        
        try:
            # Utiliser le tri topologique pour affiner l'ordre au sein de la track
            track_frame_indices = list(range(len(valid_track_frames)))
            
            # Sauvegarder temporairement les frames pour la fonction de réordonnancement
            if output_dir:
                temp_dir = os.path.join(output_dir, 'temp_topological')
                os.makedirs(temp_dir, exist_ok=True)
                
                # Sauvegarder les frames de la track
                track_frame_paths = []
                for i, idx in enumerate(valid_track_frames):
                    path = os.path.join(temp_dir, f'track_frame_{i:04d}.jpg')
                    cv2.imwrite(path, valid_frames[valid_indices.index(idx)])
                    track_frame_paths.append(path)
                    
                # Appliquer le tri topologique
                reordered_paths = AdvancedFrameReorderer.reorder_by_topological_sort(
                    track_frame_paths, track_features, similarity_threshold=0.6
                )
                
                # Récupérer l'ordre
                topological_order = []
                for path in reordered_paths:
                    base_name = os.path.basename(path)
                    index = int(base_name.split('_')[2].split('.')[0])
                    topological_order.append(index)
                    
                reordered_track_frames = [valid_track_frames[i] for i in topological_order]
            else:
                # Fallback si pas de répertoire de sortie: ordre original
                reordered_track_frames = valid_track_frames
        
        except Exception as e:
            self.logger.warning(f"Échec du réordonnancement topologique: {e}")
            reordered_track_frames = valid_track_frames
        
        # Appliquer un lissage local pour corriger d'éventuelles inversions
        track_info = [
            {'frame_idx': idx, 'score': i, 'track_id': best_track_id} 
            for i, idx in enumerate(reordered_track_frames)
        ]
        
        smoothed_indices = AutoTrackReorderUtils._smooth_local_inversions(
            track_info, window_size=5, jump_threshold=0.2
        )
        
        # Créer la liste finale des frames réordonnées
        reordered_frames = []
        
        # D'abord les frames de la track dominante dans l'ordre corrigé
        for idx in smoothed_indices:
            frame_index = valid_indices.index(idx) if idx in valid_indices else -1
            if frame_index >= 0:
                reordered_frames.append(valid_frames[frame_index])
        
        # Puis ajouter les frames restantes qui ne font pas partie de la track
        remaining_indices = [i for i, frame in enumerate(valid_frames) 
                            if valid_indices[i] not in smoothed_indices]
        for i in remaining_indices:
            reordered_frames.append(valid_frames[i])
        
        if output_dir:
            # Visualiser le réordonnancement
            if self.config.get('save_all_visualizations', True):
                output_viz_dir = os.path.join(output_dir, 'fused_reordering')
                os.makedirs(output_viz_dir, exist_ok=True)
                
                plt.figure(figsize=(12, 6))
                plt.scatter(range(len(smoothed_indices)), smoothed_indices, alpha=0.7)
                plt.plot(range(len(smoothed_indices)), smoothed_indices, 'r-', alpha=0.5)
                plt.xlabel('Nouvelle position')
                plt.ylabel('Position originale dans les frames valides')
                plt.title(f'Réordonnancement hybride (Track {best_track_id})')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(output_viz_dir, 'fused_reordering_map.png'))
                plt.close()
                
                # Sauvegarder des visualisations des frames
                visualization_dir = os.path.join(output_viz_dir, 'frames')
                os.makedirs(visualization_dir, exist_ok=True)
                
                for i, frame in enumerate(reordered_frames[:min(20, len(reordered_frames))]):
                    viz_frame = frame.copy()
                    cv2.putText(viz_frame, f"Position: {i}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    cv2.imwrite(os.path.join(visualization_dir, f'reordered_{i:04d}.jpg'), viz_frame)
        
        return reordered_frames
    
    def create_processing_dirs(self, video_name: str) -> Dict[str, str]:
        """
        Crée la structure de répertoires pour le traitement d'une vidéo
        
        Args:
            video_name: Nom de la vidéo (sans extension)
            
        Returns:
            Dictionnaire des chemins des répertoires
        """
        # Timestamp pour éviter les collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Répertoire principal pour cette vidéo
        video_dir = os.path.join(self.config['output_root'], 'processed', f"{video_name}_{timestamp}")
        os.makedirs(video_dir, exist_ok=True)
        
        # Sous-répertoires pour les différentes étapes
        dirs = {
            'root': video_dir,
            'frames': os.path.join(video_dir, 'frames'),
            'features': os.path.join(video_dir, 'features'),
            'outliers': os.path.join(video_dir, 'outliers'),
            'reordering': os.path.join(video_dir, 'reordering'),
            'tracking': os.path.join(video_dir, 'tracking'),
            'analysis': os.path.join(video_dir, 'analysis'),
            'results': os.path.join(video_dir, 'results')
        }
        
        # Créer tous les répertoires
        for dir_path in dirs.values():
            os.makedirs(dir_path, exist_ok=True)
        
        self.logger.info(f"Structure de répertoires créée pour {video_name}")
        return dirs
    
    def extract_frames(self, video_path: str, output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Extrait les frames d'une vidéo
        
        Args:
            video_path: Chemin de la vidéo
            output_dir: Répertoire de sortie pour les frames et analyses
            
        Returns:
            Liste des frames extraites (arrays numpy)
        """
        self.logger.info(f"Extraction des frames de: {video_path}")
        
        if output_dir is None:
            output_dir = os.path.join(self.config['output_root'], 'frames')
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Utiliser VideoUtils pour l'extraction avec analyse
            frame_paths = VideoUtils.extract_frames(
                video_path,
                output_dir=output_dir,
                max_frames=self.config['max_frames'],
                save_analysis=self.config['save_all_visualizations'],
                verbose=True
            )
            
            # Charger les frames en mémoire
            frames = []
            
            # Utiliser tqdm si disponible
            if TQDM_AVAILABLE:
                iterator = tqdm(frame_paths, desc="Chargement des frames")
            else:
                iterator = frame_paths
                self.logger.info(f"Chargement de {len(frame_paths)} frames...")
            
            for frame_path in iterator:
                frame = cv2.imread(frame_path)
                if frame is not None:
                    frames.append(frame)
                else:
                    self.logger.warning(f"Impossible de lire la frame: {frame_path}")
            
            self.logger.info(f"Extraction terminée: {len(frames)}/{len(frame_paths)} frames chargées")
            
            # Analyser la qualité si demandé
            if self.config['save_all_visualizations']:
                quality_dir = os.path.join(output_dir, 'quality_analysis')
                VideoUtils.analyze_video_quality(frame_paths, output_dir=quality_dir)
                self.logger.info(f"Analyse de qualité sauvegardée dans: {quality_dir}")
            
            return frames
        
        except Exception as e:
            self.logger.error(f"Échec de l'extraction des frames: {e}")
            self.logger.debug(traceback.format_exc())
            return []
    
    def extract_features(self, frames: List[np.ndarray], output_dir: Optional[str] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Extrait les caractéristiques des frames pour la détection d'anomalies
        
        Args:
            frames: Liste des frames (arrays numpy)
            output_dir: Répertoire de sortie pour les visualisations
            
        Returns:
            Tuple (features, frame_names)
        """
        self.logger.info(f"Extraction des caractéristiques pour {len(frames)} frames")
        
        if not frames or len(frames) == 0:
            self.logger.warning("Aucune frame à analyser.")
            return np.array([]), []
        
        if output_dir is None:
            output_dir = os.path.join(self.config['output_root'], 'features')
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Sauvegarder temporairement les frames pour l'extraction
            temp_frames_dir = os.path.join(output_dir, 'temp_frames')
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            frame_paths = []
            for i, frame in enumerate(frames):
                frame_path = os.path.join(temp_frames_dir, f'frame_{i:04d}.jpg')
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            # Extraire les caractéristiques avec l'outil dédié
            features, frame_names = AdvancedFeatureExtractor.extract_frame_features(
                frame_paths,
                verbose=self.config['verbose_feature_extraction'],
                output_dir=output_dir
            )
            
            # Nettoyer les fichiers temporaires si non nécessaires
            if not self.config['save_all_visualizations']:
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
            self.logger.info(f"Extraction de caractéristiques terminée: {features.shape}")
            return features, frame_names
        
        except Exception as e:
            self.logger.error(f"Échec de l'extraction des caractéristiques: {e}")
            self.logger.debug(traceback.format_exc())
            
            # Essayer une méthode d'extraction simplifiée en cas d'échec
            self.logger.info("Tentative d'extraction simplifiée...")
            return self._extract_simple_features(frames, output_dir)
    
    def _extract_simple_features(self, frames: List[np.ndarray], output_dir: str) -> Tuple[np.ndarray, List[str]]:
        """
        Méthode d'extraction de caractéristiques simplifiée (fallback)
        
        Args:
            frames: Liste des frames
            output_dir: Répertoire de sortie
            
        Returns:
            Tuple (features, frame_names)
        """
        features = []
        frame_names = []
        
        for i, frame in enumerate(frames):
            try:
                # Convertir en niveaux de gris
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Calculer des caractéristiques simples
                avg_pixel = np.mean(gray)
                std_pixel = np.std(gray)
                
                # Histogramme simplifié
                hist = cv2.calcHist([gray], [0], None, [16], [0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                # Créer un vecteur de caractéristiques
                feature_vector = np.concatenate(([avg_pixel, std_pixel], hist))
                
                features.append(feature_vector)
                frame_names.append(f'frame_{i:04d}')
            
            except Exception as e:
                self.logger.warning(f"Échec de l'extraction pour la frame {i}: {e}")
                # Utiliser un vecteur de zéros si l'extraction échoue
                features.append(np.zeros(18))  # 16 bins + 2 autres caractéristiques
                frame_names.append(f'frame_{i:04d}')
        
        features_array = np.array(features)
        
        # Sauvegarder une visualisation simple si demandé
        if self.config['save_all_visualizations'] and len(features) > 0:
            try:
                # Créer un répertoire pour les visualisations
                vis_dir = os.path.join(output_dir, 'simple_visualizations')
                os.makedirs(vis_dir, exist_ok=True)
                
                # Heatmap des caractéristiques
                plt.figure(figsize=(12, 8))
                sns.heatmap(features_array, cmap='viridis')
                plt.title('Heatmap des caractéristiques')
                plt.xlabel('Caractéristique')
                plt.ylabel('Frame')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'features_heatmap.png'))
                plt.close()
                
                # Évolution de la luminosité et du contraste
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.plot(features_array[:, 0])
                plt.title('Évolution de la luminosité')
                plt.xlabel('Frame')
                plt.ylabel('Luminosité moyenne')
                
                plt.subplot(1, 2, 2)
                plt.plot(features_array[:, 1])
                plt.title('Évolution du contraste')
                plt.xlabel('Frame')
                plt.ylabel('Écart-type')
                
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'luminosite_contraste.png'))
                plt.close()
            
            except Exception as e:
                self.logger.warning(f"Échec de la création des visualisations: {e}")
        
        self.logger.info(f"Extraction simplifiée terminée: {len(features)} vecteurs")
        return features_array, frame_names
    
    def detect_outliers(self, features: np.ndarray, output_dir: Optional[str] = None) -> Set[int]:
        """
        Détecte les frames aberrantes (outliers)
        
        Args:
            features: Matrice de caractéristiques
            output_dir: Répertoire de sortie pour les visualisations
            
        Returns:
            Ensemble des indices des outliers
        """
        self.logger.info("Détection des outliers...")
        
        if features is None or features.size == 0:
            self.logger.warning("Aucune caractéristique à analyser.")
            return set()
        
        if output_dir is None:
            output_dir = os.path.join(self.config['output_root'], 'outliers')
        
        os.makedirs(output_dir, exist_ok=True)
        
        try:
            # Utiliser le détecteur d'outliers avancé
            outlier_indices = AdvancedOutlierDetector.detect_outliers_ensemble(
                features,
                methods=self.config['outlier_detection_methods'],
                output_dir=output_dir,
                visualize=self.config['visualize_outliers'] or self.config['save_all_visualizations']
            )
            
            self.logger.info(f"Outliers détectés: {len(outlier_indices)}/{features.shape[0]} frames")
            
            # Sauvegarder la liste des outliers
            if outlier_indices:
                outliers_list = sorted(list(outlier_indices))
                with open(os.path.join(output_dir, 'outliers_list.json'), 'w') as f:
                    json.dump({'outlier_indices': outliers_list}, f, indent=2)
                
                self.logger.debug(f"Indices des outliers: {outliers_list}")
            
            return outlier_indices
        
        except Exception as e:
            self.logger.error(f"Échec de la détection d'outliers: {e}")
            self.logger.debug(traceback.format_exc())
            return set()
        

    def _save_frames_temp(self, frames: List[np.ndarray], output_dir: Optional[str] = None) -> List[str]:
        """
        Sauvegarde temporairement les frames pour les méthodes qui nécessitent des chemins de fichiers
        
        Args:
            frames: Liste des frames à sauvegarder
            output_dir: Répertoire de sortie (utilise un répertoire temporaire si non spécifié)
            
        Returns:
            Liste des chemins de fichiers des frames sauvegardées
        """
        if output_dir is None:
            temp_dir = os.path.join(self.config['output_root'], 'temp')
        else:
            temp_dir = os.path.join(output_dir, 'temp')
        
        os.makedirs(temp_dir, exist_ok=True)
        
        frame_paths = []
        for i, frame in enumerate(frames):
            frame_path = os.path.join(temp_dir, f'frame_{i:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frame_paths.append(frame_path)
        
        return frame_paths
    
    def _load_frames_from_paths(self, frame_paths: List[str]) -> List[np.ndarray]:
        """
        Charge les frames à partir d'une liste de chemins de fichiers
        
        Args:
            frame_paths: Liste des chemins de fichiers des frames
            
        Returns:
            Liste des frames chargées
        """
        frames = []
        for path in frame_paths:
            frame = cv2.imread(path)
            if frame is not None:
                frames.append(frame)
            else:
                self.logger.warning(f"Impossible de lire l'image: {path}")
        
        return frames
    
    def _visualize_bbox_evolution(self, track_data: Dict[str, Any], track_id: int, 
                            frame_indices: List[int], output_dir: str):
        """
        Visualise l'évolution de la taille des bounding boxes pour un track
        
        Args:
            track_data: Données du track
            track_id: ID du track
            frame_indices: Indices des frames valides
            output_dir: Répertoire de sortie pour les visualisations
        """
        try:
            # Calculer l'évolution de la taille des bboxes
            area_evolution = []
            ordered_frame_indices = []
            
            for frame_idx in frame_indices:
                if frame_idx in track_data['frames']:
                    track_idx = track_data['frames'].index(frame_idx)
                    if track_idx < len(track_data['bboxes']):
                        bbox = track_data['bboxes'][track_idx]
                        x1, y1, x2, y2 = bbox
                        area = (x2 - x1) * (y2 - y1)
                        area_evolution.append(area)
                        ordered_frame_indices.append(frame_idx)
            
            # Créer la visualisation
            import matplotlib.pyplot as plt
            plt.figure(figsize=(12, 6))
            plt.plot(area_evolution)
            plt.title(f'Évolution de la taille de la bbox (Track ID {track_id})')
            plt.xlabel('Position dans la séquence réordonnée')
            plt.ylabel('Surface de la bbox (pixels²)')
            plt.grid(True, alpha=0.3)
            
            # Sauvegarder le graphique
            bbox_evolution_dir = os.path.join(output_dir, 'bbox_evolution')
            os.makedirs(bbox_evolution_dir, exist_ok=True)
            plt.savefig(os.path.join(bbox_evolution_dir, 'bbox_area_evolution.png'))
            
            # Sauvegarder les données d'évolution
            with open(os.path.join(bbox_evolution_dir, 'ordered_frames.txt'), 'w') as f:
                f.write(f"Track ID: {track_id}\n")
                f.write("Index, Frame, Area\n")
                for i, (frame_idx, area) in enumerate(zip(ordered_frame_indices, area_evolution)):
                    f.write(f"{i}, {frame_idx}, {area}\n")
            
            plt.close()
        except Exception as e:
            self.logger.warning(f"[VISUALISATION] Erreur: {e}")

    
    def reorder_frames(self, 
                  frames: List[np.ndarray], 
                  features: np.ndarray, 
                  outlier_indices: Set[int], 
                  method: Optional[str] = None,
                  output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Réordonne les frames avec correction de l'orientation des tracks
        
        Args:
            frames: Liste des frames à réordonner
            features: Matrice de caractéristiques pour ces frames
            outlier_indices: Ensemble des indices des frames aberrantes
            method: Méthode de réordonnancement à utiliser
            output_dir: Répertoire de sortie pour les visualisations
            
        Returns:
            Liste des frames réordonnées
        """
        if not method:
            method = self.config['reordering_method']
        
        self.logger.info(f"Réordonnancement avec méthode : {method}")
        
        if not frames or len(frames) == 0:
            self.logger.warning("Aucune frame à réordonner.")
            return frames
        
        # Créer un ensemble d'indices de frames valides (non aberrantes)
        valid_indices = list(set(range(len(frames))) - set(outlier_indices))
        valid_frames = [frames[i] for i in valid_indices]
        
        if len(valid_frames) < self.config.get('min_frames_for_processing', 5):
            self.logger.warning(f"Nombre insuffisant de frames valides: {len(valid_frames)}")
            return frames
        
        try:
            # Vérification supplémentaire pour le tracking
            if method in ['tracking', 'fused'] and not YOLO_AVAILABLE:
                self.logger.warning("YOLO non disponible, fallback vers feature matching")
                method = 'feature_matching'
            
            # Réordonnancement selon la méthode choisie
            if method == 'tracking':
                reordered_frames = self._tracking_reorder_with_orientation_check(valid_frames, output_dir)
            elif method == 'feature_matching':
                frame_paths = self._save_frames_temp(valid_frames, output_dir)
                reordered_paths = AdvancedFrameReorderer.advanced_frame_reordering(
                    frame_paths, 
                    features[valid_indices], 
                    method='feature_matching',
                    output_dir=output_dir
                )
                reordered_frames = self._load_frames_from_paths(reordered_paths)
            elif method == 'topological':
                frame_paths = self._save_frames_temp(valid_frames, output_dir)
                reordered_paths = AdvancedFrameReorderer.reorder_by_topological_sort(
                    frame_paths, 
                    features[valid_indices]
                )
                reordered_frames = self._load_frames_from_paths(reordered_paths)
            elif method == 'optical_flow':
                reordered_frames = AdvancedFrameReorderer.reorder_by_optical_flow_continuity(valid_frames)
            elif method == 'fused':
                # Appeler la méthode de fusion
                tracking_results = self.person_tracker.detect_and_track_person(
                    valid_frames, output_dir=os.path.join(output_dir, 'tracking') if output_dir else None
                )
                person_tracks = tracking_results.get('person_tracks', {})
                reordered_frames = self.reorder_fused(
                    valid_frames, features[valid_indices], 
                    person_tracks=person_tracks,
                    output_dir=output_dir
                )
            else:
                self.logger.warning(f"Méthode non reconnue : {method}")
                return valid_frames
            
            # Vérification de la cohérence du réordonnancement
            if not reordered_frames or len(reordered_frames) < len(valid_frames):
                self.logger.warning("Réordonnancement incomplet, utilisation des frames valides")
                return valid_frames
            
            return reordered_frames
        
        except Exception as e:
            self.logger.error(f"Échec du réordonnancement : {e}")
            self.logger.debug(traceback.format_exc())
            
            # Essayer la méthode de secours
            fallback_method = self.config.get('fallback_reordering', 'feature_matching')
            if fallback_method != method:
                self.logger.info(f"Tentative avec méthode de secours : {fallback_method}")
                return self.reorder_frames(
                    frames, features, outlier_indices, 
                    method=fallback_method, 
                    output_dir=output_dir
                )
            
            return valid_frames
            
    def _tracking_reorder_with_orientation_check(self, frames: List[np.ndarray], output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Réordonne les frames en utilisant le tracking et en corrigeant l'orientation
        
        Args:
            frames: Liste des frames à réordonner
            output_dir: Répertoire de sortie pour les visualisations
            
        Returns:
            Liste des frames réordonnées
        """
        # Détecter et tracker les personnes
        tracking_results = self.person_tracker.detect_and_track_person(frames, output_dir=output_dir)
        person_tracks = tracking_results.get('person_tracks', {})
        
        if not person_tracks:
            self.logger.warning("Aucune track détectée pour le réordonnancement.")
            return frames
        
        # Trouver le track principal
        best_track_id = max(person_tracks.keys(), key=lambda k: len(person_tracks[k]['frames']))
        best_track = person_tracks[best_track_id]
        
        # Calculer les aires des bounding boxes
        bbox_areas = [
            (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) 
            for bbox in best_track['bboxes']
        ]
        
        # Vérifier si l'ordre doit être inversé
        if len(bbox_areas) > 5:
            # Comparer le début et la fin de la séquence
            start_area = bbox_areas[0]
            end_area = bbox_areas[-1]
            
            # Si la taille diminue significativement, inverser l'ordre
            if start_area > end_area * 1.5:
                self.logger.info(f"[REORDER] Track ID {best_track_id} inversé - de grandes boxes au début")
                best_track['frames'].reverse()
                best_track['bboxes'].reverse()
        
        # Vérifier que les indices de frames sont valides
        valid_indices = []
        for idx in best_track['frames']:
            if 0 <= idx < len(frames):
                valid_indices.append(idx)
            else:
                self.logger.warning(f"Index de frame invalide ignoré: {idx}")
        
        # Si aucun indice valide, retourner les frames d'origine
        if not valid_indices:
            self.logger.warning("Aucun indice de frame valide dans le track principal")
            return frames
        
        # Reconstruire les frames réordonnées
        reordered_frames = [frames[idx] for idx in valid_indices]
        
        # Log des informations de réordonnancement
        self.logger.info(f"Réordonnancement via track_id {best_track_id}: {len(reordered_frames)}/{len(frames)} frames")
        
        # Visualiser l'évolution de la taille des bounding boxes si demandé
        if self.config.get('save_all_visualizations', True) and output_dir:
            self._visualize_bbox_evolution(best_track, best_track_id, valid_indices, output_dir)
        
        return reordered_frames
        
        # Fonction pour évaluer la cohérence du réordonnancement
        def evaluate_reordering_coherence(original_frames, reordered_frames):
            """
            Calcule un score de cohérence entre les frames originales et réordonnées
            """
            def frame_similarity(frame1, frame2):
                # Convertir en niveaux de gris
                gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
                gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
                
                # Calcul de la similarité structurelle
                from skimage.metrics import structural_similarity as ssim
                return ssim(gray1, gray2)
            
            # Calculer la similarité entre frames adjacentes
            original_similarities = [
                frame_similarity(original_frames[i], original_frames[i+1]) 
                for i in range(len(original_frames)-1)
            ]
            
            reordered_similarities = [
                frame_similarity(reordered_frames[i], reordered_frames[i+1]) 
                for i in range(len(reordered_frames)-1)
            ]
            
            # Score de cohérence basé sur la similarité moyenne
            import numpy as np
            original_mean = np.mean(original_similarities)
            reordered_mean = np.mean(reordered_similarities)
            
            return 1 - abs(original_mean - reordered_mean)
        
        # Liste des méthodes à essayer
        methods_to_try = [
            method,  # Méthode principale
            self.config.get('fallback_reordering', 'feature_matching'),  # Méthode de secours
            'tracking',  # Méthode par défaut
            'feature_matching'  # Dernière option
        ]
        
        best_reordered_frames = valid_frames
        best_coherence_score = 0
        
        for attempt_method in methods_to_try:
            try:
                # Vérifier si la méthode existe
                if attempt_method not in reordering_strategies:
                    self.logger.warning(f"Méthode de réordonnancement non reconnue: {attempt_method}")
                    continue
                
                # Appeler la stratégie de réordonnancement
                reordering_func = reordering_strategies[attempt_method]
                reordered_frames = reordering_func(
                    frame_paths, 
                    features[valid_indices], 
                    output_dir
                )
                
                # Convertir les chemins de frames en images
                reordered_frame_images = [cv2.imread(path) for path in reordered_frames]
                
                # Évaluer la cohérence
                coherence_score = evaluate_reordering_coherence(valid_frames, reordered_frame_images)
                
                # Mettre à jour si meilleure cohérence
                if coherence_score > best_coherence_score:
                    best_reordered_frames = reordered_frame_images
                    best_coherence_score = coherence_score
                    
                    self.logger.info(f"Méthode {attempt_method} sélectionnée avec un score de cohérence de {coherence_score:.2f}")
                
                # Arrêter si cohérence satisfaisante
                if coherence_score > 0.8:
                    break
            
            except Exception as e:
                self.logger.warning(f"Échec du réordonnancement avec {attempt_method}: {e}")
        
        # Visualiser le réordonnancement
        if self.config.get('save_all_visualizations', True):
            try:
                # Créer une comparaison visuelle
                plt.figure(figsize=(15, 6))
                plt.subplot(121)
                plt.title('Ordre Original')
                plt.plot(range(len(valid_frames)), range(len(valid_frames)), 'b-')
                
                plt.subplot(122)
                plt.title('Ordre Réordonnancé')
                plt.plot(range(len(best_reordered_frames)), 
                        [valid_frames.index(frame) for frame in best_reordered_frames], 
                        'r-')
                
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, 'reordering_comparison.png'))
                plt.close()
            except Exception as e:
                self.logger.warning(f"Erreur de visualisation : {e}")
        
        # Log et retour
        self.logger.info(f"Réordonnancement terminé avec score de cohérence : {best_coherence_score:.2f}")
        return best_reordered_frames or valid_frames

    
    def create_video(self, 
                   frames: List[np.ndarray], 
                   output_path: str, 
                   fps: float = 30.0,
                   create_comparison: bool = True) -> bool:
        """
        Crée une vidéo à partir des frames spécifiées
        
        Args:
            frames: Liste des frames
            output_path: Chemin de sortie pour la vidéo
            fps: Images par seconde
            create_comparison: Créer également une vidéo de comparaison
            
        Returns:
            True si la vidéo a été créée avec succès
        """
        self.logger.info(f"Création de la vidéo: {output_path}")
        
        if not frames or len(frames) == 0:
            self.logger.error("Aucune frame à enregistrer.")
            return False
        
        try:
            # Créer le dossier de sortie si nécessaire
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Définir les dimensions de la vidéo
            height, width = frames[0].shape[:2]
            
            # Initialiser le writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            
            # Écrire chaque frame
            if TQDM_AVAILABLE:
                iterator = tqdm(frames, desc="Création de la vidéo")
            else:
                iterator = frames
                self.logger.info(f"Écriture de {len(frames)} frames...")
            
            for frame in iterator:
                writer.write(frame)
            
            writer.release()
            self.logger.info(f"Vidéo enregistrée: {output_path}")
            
            # Créer une vidéo d'aperçu à bas débit pour la visualisation rapide
            preview_path = os.path.splitext(output_path)[0] + '_preview.mp4'
            preview_writer = cv2.VideoWriter(preview_path, fourcc, fps, (width//2, height//2))
            
            for frame in frames:
                # Redimensionner pour l'aperçu
                resized = cv2.resize(frame, (width//2, height//2))
                preview_writer.write(resized)
            
            preview_writer.release()
            self.logger.info(f"Aperçu enregistré: {preview_path}")
            
            return True
        
        except Exception as e:
            self.logger.error(f"Échec de la création de la vidéo: {e}")
            self.logger.debug(traceback.format_exc())
            return False
    
    def create_comparison_video(self, 
                             original_frames: List[np.ndarray], 
                             processed_frames: List[np.ndarray], 
                             output_path: str, 
                             fps: float = 30.0) -> bool:
        """
        Crée une vidéo de comparaison côte à côte
        
        Args:
            original_frames: Liste des frames originales
            processed_frames: Liste des frames traitées
            output_path: Chemin de sortie pour la vidéo
            fps: Images par seconde
            
        Returns:
            True si la vidéo a été créée avec succès
        """
        self.logger.info(f"Création de la vidéo de comparaison: {output_path}")
        
        if not original_frames or not processed_frames:
            self.logger.error("Frames insuffisantes pour la comparaison.")
            return False
        
        try:
            # Créer le dossier de sortie si nécessaire
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Récupérer les dimensions
            height, width = original_frames[0].shape[:2]
            
            # Initialiser le writer pour la vidéo côte à côte
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
            
            # Nombre de frames à traiter
            n_frames = min(len(original_frames), len(processed_frames))
            
            # Écrire chaque paire de frames
            if TQDM_AVAILABLE:
                iterator = tqdm(range(n_frames), desc="Création de la comparaison")
            else:
                iterator = range(n_frames)
                self.logger.info(f"Création de la vidéo de comparaison ({n_frames} frames)...")
            
            for i in iterator:
                # Préparer les frames
                orig_frame = original_frames[i].copy()
                proc_frame = processed_frames[i].copy()
                
                # Ajouter des légendes
                cv2.putText(orig_frame, "Original", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(proc_frame, "Processed", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Combiner côte à côte
                combined = np.hstack((orig_frame, proc_frame))
                
                # Écrire la frame combinée
                writer.write(combined)
            
            writer.release()
            self.logger.info(f"Vidéo de comparaison enregistrée: {output_path}")
            return True
        
        except Exception as e:
            self.logger.error(f"Échec de la création de la vidéo de comparaison: {e}")
            self.logger.debug(traceback.format_exc())
            return False
        

    def analyze_video_objects(self, frames: List[np.ndarray], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Analyse les objets dans les frames de la vidéo
        
        Args:
            frames: Liste des frames (arrays numpy)
            output_dir: Répertoire pour les résultats
            
        Returns:
            Dictionnaire des résultats d'analyse
        """
        self.logger.info(f"Analyse des objets dans {len(frames)} frames...")
        
        if not self.person_tracker:
            self.logger.warning("Tracker de personnes non disponible")
            return {
                'status': 'error',
                'message': 'Tracker non disponible',
                'total_frames': len(frames)
            }
        
        if not frames or len(frames) == 0:
            self.logger.warning("Aucune frame à analyser")
            return {
                'status': 'error',
                'message': 'Aucune frame à analyser',
                'total_frames': 0
            }
        
        try:
            # Définir le répertoire de sortie
            if output_dir is None:
                output_dir = os.path.join(self.config['output_root'], 'object_analysis')
            
            os.makedirs(output_dir, exist_ok=True)
            
            # Sauvegarder les frames temporairement pour l'analyse
            temp_frames_dir = os.path.join(output_dir, 'temp_frames')
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            frame_paths = []
            for i, frame in enumerate(frames):
                frame_path = os.path.join(temp_frames_dir, f'frame_{i:04d}.jpg')
                cv2.imwrite(frame_path, frame)
                frame_paths.append(frame_path)
            
            # Exécuter la détection YOLO avec visualisations et export CSV
            self.logger.info("Visualisation des détections YOLO et export CSV...")
            yolo_viz_paths = self.person_tracker.visualize_yolo_detections(
                frame_paths, 
                output_dir=os.path.join(output_dir, 'yolo_detections'),
                save_all_frames=True,  # Sauvegarder toutes les frames pour l'analyse complète
                export_csv=True        # Activer l'export CSV des détections
            )
            
            # Charger les statistiques
            stats_path = os.path.join(output_dir, 'yolo_detections', 'detection_statistics.json')
            csv_path = os.path.join(output_dir, 'yolo_detections', 'yolo_detections.csv')
            
            if os.path.exists(stats_path):
                with open(stats_path, 'r') as f:
                    detection_stats = json.load(f)
            else:
                self.logger.warning("Statistiques de détection non disponibles")
                detection_stats = {
                    'total_frames': len(frames),
                    'frames_with_detections': 0,
                    'total_detections': 0,
                    'object_counts': {}
                }
            
            # Ajouter les chemins de visualisation et CSV aux statistiques
            detection_stats['visualization_paths'] = yolo_viz_paths
            detection_stats['csv_export_path'] = csv_path if os.path.exists(csv_path) else None
            detection_stats['status'] = 'success'
            
            # Log d'information sur le CSV
            if os.path.exists(csv_path):
                self.logger.info(f"Détections exportées au format CSV: {csv_path}")
            
            # Nettoyer les fichiers temporaires si non nécessaires
            if not self.config['save_all_visualizations']:
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
            # Sauvegarder un rapport d'analyse complet
            report_path = os.path.join(output_dir, 'analysis_report.json')
            with open(report_path, 'w') as f:
                json.dump(detection_stats, f, indent=2)
            
            self.logger.info(f"Analyse des objets terminée: {detection_stats['total_detections']} détections")
            
            return detection_stats
        
        except Exception as e:
            self.logger.error(f"Échec de l'analyse des objets: {e}")
            self.logger.debug(traceback.format_exc())
            
            return {
                'status': 'error',
                'message': str(e),
                'total_frames': len(frames),
                'frames_with_detections': 0,
                'total_detections': 0,
                'object_counts': {}
            }
        
    def process_video(self, input_path: str, output_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Traite la vidéo d'entrée et produit une vidéo corrigée
        
        Args:
            input_path: Chemin de la vidéo d'entrée
            output_path: Chemin de sortie pour la vidéo traitée
                
        Returns:
            Métadonnées du traitement
        """
        self.logger.info(f"\n=== Début du traitement de {input_path} ===")
        metadata = {}
        
        # Vérifier que la vidéo existe
        if not os.path.exists(input_path):
            error_msg = f"La vidéo {input_path} n'existe pas."
            self.logger.error(error_msg)
            return {'status': 'error', 'message': error_msg}
        
        try:
            # Créer une structure de répertoires pour ce traitement
            video_name = os.path.splitext(os.path.basename(input_path))[0]
            dirs = self.create_processing_dirs(video_name)
            
            # 1. Extraire les frames
            self.logger.info("Étape 1: Extraction des frames...")
            frames = self.extract_frames(input_path, output_dir=dirs['frames'])
            
            if not frames or len(frames) == 0:
                error_msg = "Aucune frame extraite."
                self.logger.error(error_msg)
                return {'status': 'error', 'message': error_msg, 'process_dirs': dirs}
            
            # Sauvegarder les frames originales pour référence
            original_frames = frames.copy()
            
            # 2. Extraire les caractéristiques
            self.logger.info(f"Étape 2: Extraction des caractéristiques pour {len(frames)} frames...")
            features, frame_names = self.extract_features(frames, output_dir=dirs['features'])
            
            # 3. Détecter les frames aberrantes
            self.logger.info("Étape 3: Détection des frames aberrantes...")
            outlier_indices = self.detect_outliers(features, output_dir=dirs['outliers'])
            
            # 4. Détecter et suivre les personnes
            self.logger.info("Étape 4: Détection et suivi des personnes...")
            tracking_results = self.person_tracker.detect_and_track_person(
                frames, 
                output_dir=os.path.join(dirs['tracking'], 'detection')
            )
            
            # 5. Réordonner les frames
            self.logger.info("Étape 5: Réordonnancement des frames...")
            
            # Utiliser le tracking pour le réordonnancement
            if self.config.get('reordering_method') == 'fused':
                person_tracks = tracking_results.get("person_tracks", {})
                
                # Vérifier si nous avons des tracks
                if not person_tracks:
                    self.logger.warning("Aucun track détecté. Fallback vers une autre méthode.")
                    reordering_method = self.config.get('fallback_reordering', 'feature_matching')
                    reordered_frames = self.reorder_frames(
                        frames, features, outlier_indices, 
                        method=reordering_method, output_dir=dirs['reordering']
                    )
                else:
                    # Convertir les clés de track en entiers si nécessaire
                    int_tracks = {int(k): v for k, v in person_tracks.items()}
                    
                    # Déterminer le track_id à utiliser
                    target_track_id = self.config.get('target_track_id')
                    if target_track_id is not None and target_track_id in int_tracks:
                        best_track_id = target_track_id
                        self.logger.info(f"[Tracking] Utilisation du track_id forcé : {best_track_id}")
                    else:
                        # Choisir le track le plus long
                        best_track_id = max(int_tracks, key=lambda k: len(int_tracks[k]['frames']))
                        self.logger.info(f"[Tracking] Track_id auto-sélectionné : {best_track_id} (le plus long)")
                    
                    # Récupérer les données du meilleur track
                    best_track = int_tracks[best_track_id]
                    
                    # Utiliser la méthode améliorée de réordonnancement
                    from frame_reorderer import AdvancedFrameReorderer
                    
                    self.logger.info(f"[REORDER] Réordonnancement via track_id {best_track_id} ({len(best_track['frames'])} frames)")
                    reordered_indices = AdvancedFrameReorderer.reorder_by_track_id(int_tracks, best_track_id)
                    
                    if reordered_indices:
                        # Réordonner les frames
                        reordered_frames = [frames[i] for i in reordered_indices if i < len(frames)]
                        self.logger.info(f"[REORDER] Réordonnancement réussi : {len(reordered_frames)} frames")
                        
                        # Visualiser l'évolution de la taille des bounding boxes
                        if self.config.get('save_all_visualizations', True):
                            try:
                                # Calculer l'évolution de la taille des bboxes
                                area_evolution = []
                                ordered_frame_indices = []
                                
                                for frame_idx in reordered_indices:
                                    if frame_idx in best_track['frames']:
                                        track_idx = best_track['frames'].index(frame_idx)
                                        if track_idx < len(best_track['bboxes']):
                                            bbox = best_track['bboxes'][track_idx]
                                            x1, y1, x2, y2 = bbox
                                            area = (x2 - x1) * (y2 - y1)
                                            area_evolution.append(area)
                                            ordered_frame_indices.append(frame_idx)
                                
                                # Créer la visualisation
                                import matplotlib.pyplot as plt
                                plt.figure(figsize=(12, 6))
                                plt.plot(area_evolution)
                                plt.title(f'Évolution de la taille de la bbox (Track ID {best_track_id})')
                                plt.xlabel('Position dans la séquence réordonnée')
                                plt.ylabel('Surface de la bbox (pixels²)')
                                plt.grid(True, alpha=0.3)
                                
                                # Sauvegarder le graphique
                                bbox_evolution_dir = os.path.join(dirs['reordering'], 'bbox_evolution')
                                os.makedirs(bbox_evolution_dir, exist_ok=True)
                                plt.savefig(os.path.join(bbox_evolution_dir, 'bbox_area_evolution.png'))
                                
                                # Sauvegarder les données d'évolution
                                with open(os.path.join(bbox_evolution_dir, 'ordered_frames.txt'), 'w') as f:
                                    f.write(f"Track ID: {best_track_id}\n")
                                    f.write("Index, Frame, Area\n")
                                    for i, (frame_idx, area) in enumerate(zip(ordered_frame_indices, area_evolution)):
                                        f.write(f"{i}, {frame_idx}, {area}\n")
                                
                                plt.close()
                            except Exception as e:
                                self.logger.warning(f"[VISUALISATION] Erreur: {e}")
                        
                        # Sauvegarder les frames réordonnées
                        reordered_dir = os.path.join(dirs['reordering'], f"track_{best_track_id}")
                        os.makedirs(reordered_dir, exist_ok=True)
                        
                        for idx, frame in enumerate(reordered_frames):
                            path = os.path.join(reordered_dir, f"reordered_{idx:04d}.jpg")
                            cv2.imwrite(path, frame)
                        
                        # Inverser l'ordre si configuré
                        if self.config.get('force_order_inversion', False):
                            self.logger.info("[REORDER] Inversion forcée de l'ordre des frames")
                            reordered_frames.reverse()
                        
                        metadata['reordered_frames'] = reordered_frames
                    else:
                        self.logger.warning(f"[REORDER] Échec du réordonnancement par track_id")
                        # Fallback vers une autre méthode
                        reordering_method = self.config.get('fallback_reordering', 'feature_matching')
                        reordered_frames = self.reorder_frames(
                            frames, features, outlier_indices, 
                            method=reordering_method, output_dir=dirs['reordering']
                        )
            else:
                # Utiliser la méthode de réordonnancement spécifiée
                reordered_frames = self.reorder_frames(
                    frames, 
                    features, 
                    outlier_indices, 
                    method=self.config['reordering_method'],
                    output_dir=dirs['reordering']
                )
            
            # Visualiser les track-id pour vérification
            track_vis_dir = os.path.join(dirs['tracking'], 'track_id_verification')
            visualized_frames = self.person_tracker.visualize_tracking_ids(
                frames, 
                tracking_results, 
                track_vis_dir
            )
            self.logger.info(f"Visualisation du tracking sauvegardée dans: {track_vis_dir}")
            
            # Vérifier si nous avons suffisamment de frames après réordonnancement
            if not reordered_frames or len(reordered_frames) < self.config.get('min_frames_for_processing', 5):
                self.logger.warning(f"Nombre insuffisant de frames après réordonnancement: {len(reordered_frames) if reordered_frames else 0}")
                self.logger.info("Utilisation des frames originales (sans outliers)")
                
                # Utiliser les frames originales sans les outliers
                valid_indices = [i for i in range(len(frames)) if i not in outlier_indices]
                reordered_frames = [frames[i] for i in valid_indices]
            
            # 6. Analyse des objets si configuré
            if self.config.get('analyze_objects', False):
                self.logger.info("Étape 6: Analyse des objets...")
                object_analysis = self.analyze_video_objects(
                    frames, 
                    output_dir=os.path.join(dirs['analysis'], 'objects')
                )
                self.logger.info(f"Analyse d'objets: {object_analysis.get('total_detections', 0)} détections")
            
            # 7. Créer la vidéo de sortie
            self.logger.info("Étape 7: Création de la vidéo de sortie...")
            
            # Définir le chemin de sortie si non spécifié
            if output_path is None:
                output_path = os.path.join(dirs['results'], f'processed_{video_name}.mp4')
            
            # Créer la vidéo traitée
            fps = self.config.get('fps', 30.0)
            
            # Utiliser les frames réordonnées déjà calculées
            frames_to_use = metadata.get('reordered_frames', reordered_frames)
            success = self.create_video(frames_to_use, output_path, fps=fps)
            
            # Créer la vidéo de comparaison
            comparison_path = os.path.join(dirs['results'], f'comparison_{video_name}.mp4')
            self.create_comparison_video(original_frames, frames_to_use, comparison_path, fps=fps)
            
            # 8. Analyser la qualité de la vidéo résultante
            self.logger.info("Étape 8: Analyse de qualité de la vidéo résultante...")
            quality_dir = os.path.join(dirs['analysis'], 'quality')
            
            # Sauvegarder temporairement les frames pour l'analyse
            temp_frames_dir = os.path.join(quality_dir, 'temp')
            os.makedirs(temp_frames_dir, exist_ok=True)
            
            reordered_frame_paths = []
            for i, frame in enumerate(frames_to_use):
                frame_path = os.path.join(temp_frames_dir, f'frame_{i:04d}.jpg')
                cv2.imwrite(frame_path, frame)
                reordered_frame_paths.append(frame_path)
            
            # Analyser la qualité
            quality_results = VideoUtils.analyze_video_quality(
                reordered_frame_paths, 
                output_dir=quality_dir
            )
            
            # Nettoyer les fichiers temporaires
            if not self.config.get('save_all_visualizations', False):
                shutil.rmtree(temp_frames_dir, ignore_errors=True)
            
            # 9. Générer le rapport d'analyse si demandé
            report_path = None
            if self.config.get('create_processing_report', True):
                self.logger.info("Étape 9: Génération du rapport d'analyse...")
                report_path = self.create_processing_report(
                    input_path,
                    output_path,
                    original_frames,
                    frames_to_use,
                    dirs,
                    tracking_results=tracking_results,
                    quality_results=quality_results
                )
                self.logger.info(f"Rapport généré: {report_path}")
            
            # Retourner les métadonnées
            metadata = {
                'status': 'success' if success else 'error',
                'input_video': input_path,
                'output_video': output_path,
                'comparison_video': comparison_path,
                'process_dirs': dirs,
                'total_frames': len(frames),
                'processed_frames': len(frames_to_use),
                'outliers_detected': len(outlier_indices) if outlier_indices else 0,
                'tracking_stats': {
                    'total_tracks': tracking_results.get('total_tracks', 0) if tracking_results else 0,
                    'person_tracks': len(tracking_results.get('person_tracks', {})) if tracking_results else 0
                },
                'report_path': report_path,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            self.logger.info(f"=== Traitement terminé avec succès: {output_path} ===")
            return metadata
        
        except Exception as e:
            self.logger.error(f"Échec du traitement de {input_path}: {e}")
            self.logger.debug(traceback.format_exc())
            return {
                'status': 'error',
                'message': str(e),
                'input_video': input_path,
                'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
    
    def create_processing_report(self, 
                               input_path: str, 
                               output_path: str,
                               original_frames: List[np.ndarray],
                               processed_frames: List[np.ndarray],
                               dirs: Dict[str, str],
                               tracking_results: Optional[Dict[str, Any]] = None,
                               quality_results: Optional[Dict[str, Any]] = None) -> str:
        """
        Crée un rapport complet du traitement de la vidéo
        
        Args:
            input_path: Chemin de la vidéo d'entrée
            output_path: Chemin de la vidéo de sortie
            original_frames: Frames originales
            processed_frames: Frames traitées
            dirs: Répertoires de traitement
            tracking_results: Résultats du tracking (optionnel)
            quality_results: Résultats de l'analyse de qualité (optionnel)
                
        Returns:
            Chemin du rapport généré
        """
        # Créer le répertoire pour le rapport
        report_dir = os.path.join(dirs['analysis'], 'report')
        os.makedirs(report_dir, exist_ok=True)
        
        # Créer un rapport JSON avec les métadonnées
        report_data = {
            'input_video': input_path,
            'output_video': output_path,
            'processing_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'processing_dirs': dirs,
            'frames_stats': {
                'original': len(original_frames),
                'processed': len(processed_frames)
            },
            'config': self.config
        }
        
        # Ajouter les informations de tracking si disponibles
        if tracking_results:
            person_tracks = tracking_results.get('person_tracks', {})
            
            # Collecter des statistiques sur les tracks
            track_stats = {
                'total_tracks': len(person_tracks),
                'average_track_length': 0,
                'max_track_length': 0,
                'continuous_coverage': 0
            }
            
            if person_tracks:
                track_lengths = []
                for track_id, track in person_tracks.items():
                    frames = track.get('frames', [])
                    track_lengths.append(len(frames))
                    
                    # Calculer la durée totale (dernier frame - premier frame)
                    if 'first_seen' in track and 'last_seen' in track:
                        duration = track['last_seen'] - track['first_seen'] + 1
                        track['duration'] = duration
                        track['continuity'] = len(frames) / duration if duration > 0 else 0
                
                if track_lengths:
                    track_stats['average_track_length'] = sum(track_lengths) / len(track_lengths)
                    track_stats['max_track_length'] = max(track_lengths)
                    
                    # Calculer la couverture continue
                    frame_coverage = set()
                    for track in person_tracks.values():
                        frame_coverage.update(track.get('frames', []))
                        
                    if len(original_frames) > 0:
                        track_stats['continuous_coverage'] = len(frame_coverage) / len(original_frames)
            
            report_data['tracking'] = {
                'track_stats': track_stats,
                'person_tracks': {k: {
                    'frames_count': len(v.get('frames', [])),
                    'first_seen': v.get('first_seen'),
                    'last_seen': v.get('last_seen'),
                    'duration': v.get('duration'),
                    'continuity': v.get('continuity')
                } for k, v in person_tracks.items()}
            }
        
        # Ajouter les résultats de l'analyse de qualité
        if quality_results:
            report_data['quality'] = quality_results
        
        # Créer une montage d'aperçu avant/après
        montage_dir = os.path.join(report_dir, 'montage')
        os.makedirs(montage_dir, exist_ok=True)
        
        # Sélectionner quelques frames pour la comparaison
        frame_count = min(len(original_frames), len(processed_frames))
        if frame_count > 5:
            indices = [0, frame_count//4, frame_count//2, 3*frame_count//4, frame_count-1]
        else:
            indices = list(range(frame_count))
        
        comparison_images = []
        
        for i, idx in enumerate(indices):
            orig_frame = original_frames[idx].copy()
            proc_frame = processed_frames[idx].copy()
            
            # Ajouter des légendes
            cv2.putText(orig_frame, f"Original (frame {idx})", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(proc_frame, f"Processed", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Combiner côte à côte
            comparison = np.hstack((orig_frame, proc_frame))
            comparison_path = os.path.join(montage_dir, f'comparison_{i:02d}.jpg')
            cv2.imwrite(comparison_path, comparison)
            comparison_images.append(comparison_path)
        
        # Créer un montage global des comparaisons
        montage_path = os.path.join(montage_dir, 'frame_comparisons.jpg')
        VideoUtils.create_frame_montage(comparison_images, montage_path, max_frames=0, columns=1)
        
        # Ajouter le chemin du montage au rapport
        report_data['montage_path'] = montage_path
        
        # Ajouter des statistiques sur les frames
        if len(original_frames) > 0 and len(processed_frames) > 0:
            # Calculer des statistiques simples sur les frames
            orig_brightness = np.mean([np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in original_frames])
            proc_brightness = np.mean([np.mean(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in processed_frames])
            
            # Calculer des statistiques de netteté
            def calculate_sharpness(frame):
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                return np.var(laplacian)
                
            orig_sharpness = np.mean([calculate_sharpness(f) for f in original_frames])
            proc_sharpness = np.mean([calculate_sharpness(f) for f in processed_frames])
            
            report_data['frames_stats']['brightness'] = {
                'original': float(orig_brightness),
                'processed': float(proc_brightness),
                'change_percentage': float((proc_brightness - orig_brightness) / orig_brightness * 100) if orig_brightness > 0 else 0
            }
            
            report_data['frames_stats']['sharpness'] = {
                'original': float(orig_sharpness),
                'processed': float(proc_sharpness),
                'change_percentage': float((proc_sharpness - orig_sharpness) / orig_sharpness * 100) if orig_sharpness > 0 else 0
            }
        
        # Sauvegarder le rapport au format JSON
        json_path = os.path.join(report_dir, 'processing_report.json')
        with open(json_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Créer un rapport humainement lisible
        txt_path = os.path.join(report_dir, 'processing_report.txt')
        with open(txt_path, 'w') as f:
            f.write("Rapport de traitement vidéo\n")
            f.write("==========================\n\n")
            f.write(f"Vidéo source: {input_path}\n")
            f.write(f"Vidéo traitée: {output_path}\n")
            f.write(f"Date de traitement: {report_data['processing_time']}\n\n")
            
            f.write("Statistiques des frames:\n")
            f.write(f"  Frames originales: {len(original_frames)}\n")
            f.write(f"  Frames traitées: {len(processed_frames)}\n")
            
            if 'brightness' in report_data['frames_stats']:
                f.write("\nStatistiques de luminosité:\n")
                f.write(f"  Original: {report_data['frames_stats']['brightness']['original']:.2f}\n")
                f.write(f"  Traité: {report_data['frames_stats']['brightness']['processed']:.2f}\n")
                f.write(f"  Variation: {report_data['frames_stats']['brightness']['change_percentage']:.2f}%\n")
            
            if 'sharpness' in report_data['frames_stats']:
                f.write("\nStatistiques de netteté:\n")
                f.write(f"  Original: {report_data['frames_stats']['sharpness']['original']:.2f}\n")
                f.write(f"  Traité: {report_data['frames_stats']['sharpness']['processed']:.2f}\n")
                f.write(f"  Variation: {report_data['frames_stats']['sharpness']['change_percentage']:.2f}%\n")
            
            # Ajouter les statistiques de tracking si disponibles
            if 'tracking' in report_data:
                track_stats = report_data['tracking']['track_stats']
                f.write("\nStatistiques de tracking:\n")
                f.write(f"  Nombre de tracks: {track_stats['total_tracks']}\n")
                f.write(f"  Longueur moyenne: {track_stats['average_track_length']:.2f} frames\n")
                f.write(f"  Longueur maximale: {track_stats['max_track_length']} frames\n")
                f.write(f"  Couverture continue: {track_stats['continuous_coverage']*100:.2f}%\n")
                
                # Détails des tracks principales
                person_tracks = report_data['tracking']['person_tracks']
                if person_tracks:
                    f.write("\nTracks principales:\n")
                    for track_id, track in sorted(person_tracks.items(), 
                                                key=lambda x: x[1].get('frames_count', 0), 
                                                reverse=True)[:5]:  # Top 5 tracks
                        f.write(f"  Track ID {track_id}:\n")
                        f.write(f"    Frames: {track.get('frames_count', 0)}\n")
                        f.write(f"    Première apparition: {track.get('first_seen', 'N/A')}\n")
                        f.write(f"    Dernière apparition: {track.get('last_seen', 'N/A')}\n")
                        f.write(f"    Durée: {track.get('duration', 'N/A')} frames\n")
                        f.write(f"    Continuité: {track.get('continuity', 0)*100:.2f}%\n")
            
            f.write("\nRépertoires de traitement:\n")
            for name, path in dirs.items():
                f.write(f"  {name}: {path}\n")
            
            f.write("\nConfiguration utilisée:\n")
            for key, value in self.config.items():
                if not isinstance(value, dict):  # Éviter les sous-dictionnaires trop détaillés
                    f.write(f"  {key}: {value}\n")
        
        # Créer des visualisations supplémentaires si des données de qualité sont disponibles
        if quality_results:
            try:
                viz_dir = os.path.join(report_dir, 'visualizations')
                os.makedirs(viz_dir, exist_ok=True)
                
                # Comparaison de luminosité
                if 'brightness' in report_data['frames_stats']:
                    plt.figure(figsize=(10, 6))
                    
                    orig_brightness = report_data['frames_stats']['brightness']['original']
                    proc_brightness = report_data['frames_stats']['brightness']['processed']
                    
                    plt.bar(['Original', 'Processed'], [orig_brightness, proc_brightness])
                    plt.title('Comparaison de luminosité moyenne')
                    plt.ylabel('Luminosité')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(viz_dir, 'brightness_comparison.png'))
                    plt.close()
                
                # Comparaison de netteté
                if 'sharpness' in report_data['frames_stats']:
                    plt.figure(figsize=(10, 6))
                    
                    orig_sharpness = report_data['frames_stats']['sharpness']['original']
                    proc_sharpness = report_data['frames_stats']['sharpness']['processed']
                    
                    plt.bar(['Original', 'Processed'], [orig_sharpness, proc_sharpness])
                    plt.title('Comparaison de netteté moyenne')
                    plt.ylabel('Netteté (Variance du Laplacien)')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(viz_dir, 'sharpness_comparison.png'))
                    plt.close()
            
            except Exception as e:
                self.logger.warning(f"Erreur lors de la création des visualisations: {e}")
        
        self.logger.info(f"Rapport de traitement généré: {txt_path}")
        return txt_path
    

# Example usage in main.py
def main():
    """
    Script principal pour le traitement de vidéo
    """
    # Configuration personnalisée du traitement vidéo
    config = {
        # Configuration générale
        'output_root': 'video_output',  # Répertoire de sortie
        'max_frames': None,  # Traiter toutes les frames
        'fps': 30,  # Fréquence d'images par seconde

        # Configuration de détection des outliers
        'outlier_detection_methods': [
            'pca',      # Analyse en Composantes Principales
            'zscore',   # Méthode du Z-score
            'isolation_forest'  # Forêt d'isolement
        ],
        'visualize_outliers': True,  # Générer des visualisations
        'save_all_visualizations': True,  # Sauvegarder toutes les visualisations

        # Configuration du tracking de personnes
        'reordering_method': 'tracking',  # Réordonner par suivi de personnes
        'yolo_model_path': 'yolov8m.pt',  # Chemin du modèle YOLO
        'tracking_confidence': 0.6,  # Seuil de confiance pour le tracking

        # Options de verbosité
        'verbose_feature_extraction': True
    }

    # Chemins des vidéos à traiter
    video_paths = [
        'corrupted_video.mp4',
        # Ajoutez d'autres chemins de vidéos si nécessaire
    ]

    # Initialiser le processeur vidéo
    processor = VideoProcessor(config)

    # Traiter chaque vidéo
    for video_path in video_paths:
        try:
            # Vérifier que la vidéo existe
            if not os.path.exists(video_path):
                print(f"[ERREUR] La vidéo {video_path} n'existe pas.")
                continue

            # Définir le chemin de sortie
            output_path = os.path.join(
                config['output_root'], 
                'processed', 
                f'processed_{os.path.basename(video_path)}'
            )

            # Traiter la vidéo
            print(f"\n--- Traitement de {video_path} ---")
            metadata = processor.process_video(video_path, output_path)

            # Afficher les métadonnées
            print("\nMétadonnées de traitement :")
            for key, value in metadata.items():
                if key != 'process_dirs':  # Ne pas afficher les chemins complets
                    print(f"{key}: {value}")

        except Exception as e:
            print(f"[ERREUR] Échec du traitement de {video_path} : {e}")
            traceback.print_exc()

    print("\nTraitement des vidéos terminé.")

if __name__ == "__main__":
    main()