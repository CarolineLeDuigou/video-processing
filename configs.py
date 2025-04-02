import os
from typing import Dict, Any, List, Optional
import json
from datetime import datetime

class VideoProcessingConfig:
    """Configuration centralisée pour le traitement vidéo avec options d'analyse avancées"""
    def __init__(self, custom_config: Dict[str, Any] = None):
        self.default_config = {
            # Seuils de détection
            'confidence_threshold': 0.85,
            'min_bbox_size': 50,
            'movement_threshold': 0.3,
            'size_change_threshold': 0.4,
            'outlier_zscore_threshold': 2.5,
            
            # Chemins et répertoires
            'output_root': 'video_processing_output',
            'temp_frames_dir': 'frames',
            'detection_dir': 'detections',
            'analysis_dir': 'analysis',
            
            # Paramètres de modèles
            'yolo_model': 'yolov8m.pt',
            'movinet_model_url': 'https://tfhub.dev/tensorflow/movinet/a2/base/kinetics-600/classification/3',
            
            # Stratégies de traitement
            'reordering_strategy': 'topological_sort',
            'outlier_detection_methods': [
                'pca', 
                'movement', 
                'clustering', 
                'histogram'
            ],
            
            # Options d'analyse et de visualisation
            'save_all_visualizations': True,            # Sauvegarder toutes les visualisations possibles
            'save_intermediate_frames': True,           # Sauvegarder les frames intermédiaires
            'create_analysis_images': True,             # Générer des images d'analyse
            'visualization_quality': 'high',            # Qualité des visualisations (low, medium, high)
            
            # Configuration d'analyse de frames
            'frame_analysis': {
                'histograms': True,                     # Générer des histogrammes de couleur
                'edge_detection': True,                 # Visualiser la détection de contours
                'motion_estimation': True,              # Visualiser l'estimation de mouvement
                'feature_visualization': True,          # Visualiser les caractéristiques extraites
                'quality_metrics': True                 # Analyser les métriques de qualité
            },
            
            # Configuration de détection d'outliers
            'outlier_analysis': {
                'save_outlier_frames': True,            # Sauvegarder les frames identifiées comme outliers
                'outlier_visualization_methods': [      # Méthodes de visualisation des outliers
                    'heatmap', 'scatter', 'tsne', 'pca'
                ],
                'create_outlier_report': True           # Générer un rapport détaillé sur les outliers
            },
            
            # Configuration de réordonnancement
            'reordering_analysis': {
                'save_reordering_map': True,            # Sauvegarder la carte de réordonnancement
                'create_comparison_montage': True,      # Créer un montage avant/après
                'save_flow_visualizations': True,       # Sauvegarder les visualisations de flux optique
                'trajectory_analysis': True             # Analyser les trajectoires
            },
            
            # Configuration de tracking
            'tracking_analysis': {
                'save_tracking_frames': True,           # Sauvegarder les frames avec tracking
                'create_trajectory_map': True,          # Créer une carte des trajectoires
                'save_tracking_statistics': True,       # Sauvegarder les statistiques de tracking
                'object_presence_timeline': True        # Créer une timeline de présence des objets
            },
            
            # Configuration des rapports
            'reports': {
                'create_processing_report': True,       # Créer un rapport global de traitement
                'report_format': 'all',                 # Format du rapport (text, json, html, all)
                'include_visualizations': True,         # Inclure des visualisations dans le rapport
                'detailed_metrics': True                # Inclure des métriques détaillées
            }
        }
        
        # Fusion de la configuration personnalisée
        self.config = {**self.default_config, **(custom_config or {})}
        
        # S'assurer que les chemins sont absolus si spécifiés
        self._normalize_paths()
        
        # Créer les répertoires nécessaires
        self._create_directories()
    
    def _normalize_paths(self):
        """Normaliser les chemins de répertoires"""
        # Convertir en chemins absolus si nécessaire
        for key in ['output_root', 'yolo_model']:
            if key in self.config and self.config[key] and not os.path.isabs(self.config[key]):
                # Vérifier si c'est un chemin vers un fichier existant
                if key == 'yolo_model' and os.path.exists(self.config[key]):
                    self.config[key] = os.path.abspath(self.config[key])
                elif key == 'output_root':
                    self.config[key] = os.path.abspath(self.config[key])
    
    def _create_directories(self):
        """Créer les répertoires nécessaires pour le traitement"""
        # Répertoires principaux
        dirs_to_create = [
            self.config['output_root'],
            os.path.join(self.config['output_root'], self.config['temp_frames_dir']),
            os.path.join(self.config['output_root'], self.config['detection_dir']),
            os.path.join(self.config['output_root'], self.config['analysis_dir'])
        ]
        
        # Répertoires d'analyse
        if self.config['save_all_visualizations']:
            # Répertoires pour l'analyse de frames
            if self.config['frame_analysis']['histograms']:
                dirs_to_create.append(os.path.join(self.config['output_root'], 'analysis', 'histograms'))
            
            if self.config['frame_analysis']['edge_detection']:
                dirs_to_create.append(os.path.join(self.config['output_root'], 'analysis', 'edge_detection'))
            
            if self.config['frame_analysis']['motion_estimation']:
                dirs_to_create.append(os.path.join(self.config['output_root'], 'analysis', 'motion'))
            
            if self.config['frame_analysis']['feature_visualization']:
                dirs_to_create.append(os.path.join(self.config['output_root'], 'analysis', 'features'))
            
            # Répertoires pour l'analyse d'outliers
            if self.config['outlier_analysis']['save_outlier_frames']:
                dirs_to_create.append(os.path.join(self.config['output_root'], 'analysis', 'outliers'))
            
            # Répertoires pour l'analyse de réordonnancement
            if self.config['reordering_analysis']['save_reordering_map']:
                dirs_to_create.append(os.path.join(self.config['output_root'], 'analysis', 'reordering'))
            
            # Répertoires pour l'analyse de tracking
            if self.config['tracking_analysis']['save_tracking_frames']:
                dirs_to_create.append(os.path.join(self.config['output_root'], 'analysis', 'tracking'))
            
            # Répertoire pour les rapports
            if self.config['reports']['create_processing_report']:
                dirs_to_create.append(os.path.join(self.config['output_root'], 'reports'))
        
        # Créer tous les répertoires nécessaires
        for dir_path in dirs_to_create:
            os.makedirs(dir_path, exist_ok=True)
    
    def get(self, key: str, default=None):
        """Récupérer une valeur de configuration"""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any):
        """Définir une valeur de configuration"""
        self.config[key] = value
        
        # Recréer les répertoires si nécessaire
        if key in ['output_root', 'temp_frames_dir', 'detection_dir', 'analysis_dir']:
            self._create_directories()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convertir la configuration en dictionnaire"""
        return self.config
    
    def save_config(self, path: Optional[str] = None) -> str:
        """
        Sauvegarder la configuration dans un fichier JSON
        
        Args:
            path: Chemin du fichier (si None, utilise output_root/config_timestamp.json)
            
        Returns:
            Chemin du fichier sauvegardé
        """
        if path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(self.config['output_root'], f'config_{timestamp}.json')
        
        # Créer le répertoire parent si nécessaire
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Sauvegarder la configuration
        with open(path, 'w') as f:
            json.dump(self.config, f, indent=2)
        
        return path
    
    @classmethod
    def load_config(cls, path: str) -> 'VideoProcessingConfig':
        """
        Charger la configuration depuis un fichier JSON
        
        Args:
            path: Chemin du fichier
            
        Returns:
            Instance de VideoProcessingConfig
        """
        with open(path, 'r') as f:
            custom_config = json.load(f)
        
        return cls(custom_config)
    
    def get_analysis_dir(self, analysis_type: str, create: bool = True) -> str:
        """
        Obtenir le chemin complet pour un type d'analyse spécifique
        
        Args:
            analysis_type: Type d'analyse (frames, outliers, reordering, tracking, etc.)
            create: Créer le répertoire s'il n'existe pas
            
        Returns:
            Chemin complet du répertoire
        """
        # Mapper les types d'analyse vers des sous-répertoires
        analysis_dirs = {
            'frames': 'frames',
            'histograms': os.path.join('analysis', 'histograms'),
            'edge_detection': os.path.join('analysis', 'edge_detection'),
            'motion': os.path.join('analysis', 'motion'),
            'features': os.path.join('analysis', 'features'),
            'outliers': os.path.join('analysis', 'outliers'),
            'reordering': os.path.join('analysis', 'reordering'),
            'tracking': os.path.join('analysis', 'tracking'),
            'reports': 'reports'
        }
        
        # Vérifier si le type d'analyse est valide
        if analysis_type not in analysis_dirs:
            raise ValueError(f"Type d'analyse inconnu: {analysis_type}")
        
        # Construire le chemin complet
        full_path = os.path.join(self.config['output_root'], analysis_dirs[analysis_type])
        
        # Créer le répertoire si demandé
        if create:
            os.makedirs(full_path, exist_ok=True)
        
        return full_path
    
    def create_processing_dirs(self, video_name: str) -> Dict[str, str]:
        """
        Crée une structure de répertoires pour le traitement d'une vidéo spécifique
        
        Args:
            video_name: Nom de la vidéo (sans extension)
            
        Returns:
            Dictionnaire des chemins des répertoires
        """
        # Timestamp pour éviter les collisions
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Répertoire principal pour cette vidéo
        video_dir = os.path.join(self.config['output_root'], 'processed', f"{video_name}_{timestamp}")
        
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
        
        # Créer les sous-répertoires d'analyse si demandé
        if self.config['save_all_visualizations']:
            analysis_subdirs = [
                'histograms',
                'edge_detection',
                'motion',
                'features',
                'quality',
                'comparison',
                'visualization'
            ]
            
            for subdir in analysis_subdirs:
                os.makedirs(os.path.join(dirs['analysis'], subdir), exist_ok=True)
        
        return dirs
    
    def __str__(self) -> str:
        """Représentation textuelle de la configuration"""
        return f"VideoProcessingConfig(output_root='{self.config['output_root']}', " \
               f"save_all_visualizations={self.config['save_all_visualizations']})"

# Example usage
def main():
    # Example custom configuration
    custom_config = {
        'output_root': 'video_output',
        'save_all_visualizations': True,
        'visualization_quality': 'high',
        'outlier_detection_methods': ['zscore', 'pca', 'isolation_forest']
    }
    
    # Create configuration instance
    config = VideoProcessingConfig(custom_config)
    
    # Print the merged configuration
    print("Configuration:")
    for key, value in config.to_dict().items():
        if not isinstance(value, dict):
            print(f"  {key}: {value}")
    
    # Test directory creation
    print(f"\nOutput directories created in: {config.get('output_root')}")
    
    # Save the configuration
    config_path = config.save_config()
    print(f"Configuration saved to: {config_path}")
    
    # Create processing directories for a specific video
    video_dirs = config.create_processing_dirs('test_video')
    print("\nProcessing directories created:")
    for name, path in video_dirs.items():
        print(f"  {name}: {path}")

if __name__ == "__main__":
    main()