import os
import cv2
import csv
import json
import numpy as np
import torch
import matplotlib.pyplot as plt
from datetime import datetime
from typing import List, Dict, Any, Optional
import scipy.signal

# Essayer d'importer YOLO (Ultralytics)
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("[WARNING] Ultralytics YOLO non disponible. Installez via : pip install ultralytics")

# Essayer d'importer DeepSORT
try:
    from deep_sort_realtime.deepsort_tracker import DeepSort
    DEEPSORT_AVAILABLE = True
except ImportError:
    DEEPSORT_AVAILABLE = False
    print("[WARNING] DeepSORT non disponible. Installez via : pip install deep-sort-realtime")


class PersonTracker:
    """
    Advanced person tracking class with detection visualization
    """

    def __init__(self, 
                 yolo_model_path: str = 'yolov8m.pt', 
                 confidence_threshold: float = 0.7,
                 tracking_threshold: float = 0.5,
                 visualization_enabled: bool = True,
                 color_similarity_threshold: float = 0.5,
                 max_frames_between_match: int = 150):
        """
        Initialize the person tracking system
        
        Args:
            yolo_model_path (str): Path to YOLO model
            confidence_threshold (float): Detection confidence threshold
            tracking_threshold (float): Tracking confidence threshold
            visualization_enabled (bool): Enable automatic visualization
            color_similarity_threshold (float): Threshold for color-based tracking
            max_frames_between_match (int): Maximum frames between matches
        """
        # Vérifier les dépendances
        if not YOLO_AVAILABLE:
            raise ImportError("YOLO n'est pas disponible. Installez-le via : pip install ultralytics")
        
        if not DEEPSORT_AVAILABLE:
            raise ImportError("DeepSORT n'est pas disponible. Installez-le via : pip install deep-sort-realtime")
        
        # Charger le modèle YOLO
        try:
            self.yolo_model = YOLO(yolo_model_path)
            print(f"YOLO model loaded: {yolo_model_path}")
        except Exception as e:
            print(f"[ERROR] Échec lors du chargement du modèle YOLO : {e}")
            raise
        
        # Initialiser le tracker DeepSORT
        self.tracker = DeepSort(max_age=5)
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.tracking_threshold = tracking_threshold
        self.visualization_enabled = visualization_enabled
        
        # Paramètres pour le tracking par histogramme de couleurs
        self.color_similarity_threshold = color_similarity_threshold
        self.max_frames_between_match = max_frames_between_match
        
        # Historique de tracking
        self.track_history = {}
        
        # Liste des classes COCO (correspondant à YOLOv8)
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 
            'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 
            'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
            'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 
            'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 
            'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 
            'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]
        
        # Génération de couleurs pour chaque classe
        np.random.seed(42)
        self.colors = np.random.randint(0, 255, size=(len(self.coco_classes), 3), dtype=np.uint8)
        
        # Dictionnaire pour couleurs des IDs de tracking
        self.track_colors = {}
        self.predicted_positions = {}  # {track_id: [x, y, vx, vy]}

    def visualize_yolo_detections(self, 
                                  frames: List[str], 
                                  output_dir: Optional[str] = None,
                                  save_all_frames: bool = True,
                                  export_csv: bool = True) -> List[str]:
        """
        Visualise les détections YOLO sur chaque frame et exporte éventuellement les résultats
        (CSV, JSON, heatmap, etc.).
        
        Args:
            frames (List[str]): Liste des chemins vers les images.
            output_dir (Optional[str]): Dossier de sortie pour enregistrer.
            save_all_frames (bool): Si True, enregistre toutes les frames en sortie.
            export_csv (bool): Si True, exporte les résultats dans un fichier CSV.

        Returns:
            List[str]: Chemins des images générées avec visualisations.
        """
        if output_dir is None:
            output_dir = os.path.join('video_output', 'yolo_detections')
        os.makedirs(output_dir, exist_ok=True)

        # Sous-dossiers
        frames_with_detections_dir = os.path.join(output_dir, 'frames_with_detections')
        detection_heatmap_dir = os.path.join(output_dir, 'detection_heatmap')
        os.makedirs(frames_with_detections_dir, exist_ok=True)
        os.makedirs(detection_heatmap_dir, exist_ok=True)

        visualization_paths = []

        # Statistiques
        detection_stats = {
            'total_frames': len(frames),
            'frames_with_detections': 0,
            'total_detections': 0,
            'object_counts': {},
            'detection_confidence': []
        }

        heatmap = None
        csv_file, csv_writer = None, None

        # Initialiser le CSV si demandé
        if export_csv:
            csv_path = os.path.join(output_dir, 'yolo_detections.csv')
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                'frame_index', 'frame_path', 'class_id', 'class_name',
                'confidence', 'x1', 'y1', 'x2', 'y2', 'width', 'height'
            ])

        print(f"Processing {len(frames)} frames with YOLO...")

        for frame_index, frame_path in enumerate(frames):
            if frame_index % 10 == 0:
                print(f"Processing frame {frame_index + 1}/{len(frames)}")

            frame = cv2.imread(frame_path)
            if frame is None:
                print(f"[WARNING] Could not read image: {frame_path}")
                continue

            if heatmap is None:
                heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)

            # Exécuter YOLO
            results = self.yolo_model(frame, conf=self.confidence_threshold)[0]
            boxes = results.boxes
            frame_has_detections = len(boxes) > 0

            if frame_has_detections:
                detection_stats['frames_with_detections'] += 1

                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])

                    class_name = self.coco_classes[cls] if cls < len(self.coco_classes) else f"class_{cls}"

                    detection_stats['total_detections'] += 1
                    detection_stats['object_counts'][class_name] = detection_stats['object_counts'].get(class_name, 0) + 1
                    detection_stats['detection_confidence'].append(conf)

                    width, height = x2 - x1, y2 - y1

                    if csv_writer:
                        csv_writer.writerow([
                            frame_index,
                            os.path.basename(frame_path),
                            cls,
                            class_name,
                            conf,
                            x1, y1, x2, y2,
                            width, height
                        ])

                    # Mettre à jour la heatmap
                    cv2.rectangle(heatmap, (x1, y1), (x2, y2), 1.0, -1)

                    color = tuple(map(int, self.colors[cls % len(self.colors)]))
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            # Décider si on sauvegarde la frame
            should_save = (
                save_all_frames or
                frame_has_detections or
                frame_index % 20 == 0 or
                frame_index < 5 or
                frame_index >= len(frames) - 5
            )

            if should_save:
                output_path = os.path.join(output_dir, f'yolo_detection_{frame_index:04d}.jpg')
                cv2.imwrite(output_path, frame)
                visualization_paths.append(output_path)

                if frame_has_detections:
                    detection_path = os.path.join(frames_with_detections_dir, os.path.basename(frame_path))
                    cv2.imwrite(detection_path, frame)

        if csv_file:
            csv_file.close()
            print(f"Detections exported to CSV: {csv_path}")

        # Sauvegarder la heatmap
        if heatmap is not None:
            if np.max(heatmap) > 0:
                heatmap_normalized = heatmap / np.max(heatmap)
            else:
                heatmap_normalized = heatmap

            heatmap_color = cv2.applyColorMap((heatmap_normalized * 255).astype(np.uint8), cv2.COLORMAP_JET)
            heatmap_path = os.path.join(detection_heatmap_dir, 'detection_heatmap.jpg')
            cv2.imwrite(heatmap_path, heatmap_color)

            if frames:
                first_frame = cv2.imread(frames[0])
                if first_frame is not None:
                    overlay = cv2.addWeighted(first_frame, 0.4, heatmap_color, 0.6, 0)
                    overlay_path = os.path.join(detection_heatmap_dir, 'detection_heatmap_overlay.jpg')
                    cv2.imwrite(overlay_path, overlay)

        # Calculs de stats
        if detection_stats['detection_confidence']:
            detection_stats['avg_confidence'] = float(np.mean(detection_stats['detection_confidence']))
            detection_stats['min_confidence'] = float(np.min(detection_stats['detection_confidence']))
            detection_stats['max_confidence'] = float(np.max(detection_stats['detection_confidence']))
        else:
            detection_stats['avg_confidence'] = 0
            detection_stats['min_confidence'] = 0
            detection_stats['max_confidence'] = 0

        # On tronque la liste pour éviter un JSON énorme
        detection_stats['detection_confidence'] = detection_stats['detection_confidence'][:100]

        # Sauvegarder les stats en JSON
        stats_path = os.path.join(output_dir, 'detection_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(detection_stats, f, indent=2)

        # Générer des graphiques
        self._create_detection_visualizations(detection_stats, output_dir)

        return visualization_paths

    def visualize_tracking_ids(self, frames, tracking_results, output_dir):
        """
        Visualise les track-id sur les images pour vérification
        
        Args:
            frames: Liste des frames originales
            tracking_results: Résultats du tracking (avec person_tracks)
            output_dir: Répertoire de sortie pour les visualisations
        """
        if 'person_tracks' not in tracking_results:
            print("Aucune donnée de tracking disponible pour la visualisation")
            return []
        
        person_tracks = tracking_results['person_tracks']
        if not person_tracks:
            print("Aucune personne trackée pour la visualisation")
            return []
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        visualized_frames = []
        
        # Traiter chaque frame
        for frame_idx, frame in enumerate(frames):
            # Copier la frame pour la visualisation
            vis_frame = frame.copy()
            tracks_in_frame = []
            
            # Vérifier quelles tracks sont présentes dans cette frame
            for track_id, track_data in person_tracks.items():
                if frame_idx in track_data['frames']:
                    # Obtenir l'index dans les données de tracking
                    track_idx = track_data['frames'].index(frame_idx)
                    
                    # Récupérer la bounding box
                    if track_idx < len(track_data['bboxes']):
                        bbox = track_data['bboxes'][track_idx]
                        x1, y1, x2, y2 = bbox
                        
                        # Obtenir une couleur persistante pour cette track
                        color = self.track_colors.get(track_id, (0, 255, 0))
                        
                        # Dessiner la bounding box
                        cv2.rectangle(vis_frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                        
                        # Afficher le track_id avec un fond pour meilleure lisibilité
                        label = f"ID: {track_id}"
                        text_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                        cv2.rectangle(vis_frame, 
                                    (int(x1), int(y1) - text_size[1] - 10),
                                    (int(x1) + text_size[0] + 10, int(y1)),
                                    color, -1)  # Remplir le rectangle
                        
                        cv2.putText(vis_frame, label, 
                                (int(x1) + 5, int(y1) - 5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                        
                        tracks_in_frame.append(track_id)
            
            # Ajouter l'information de frame
            cv2.putText(vis_frame, f"Frame: {frame_idx}", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
            # Ajouter la liste des track_id présents
            cv2.putText(vis_frame, f"Tracks: {tracks_in_frame}", 
                    (10, vis_frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Sauvegarder la frame
            output_path = os.path.join(output_dir, f"track_vis_{frame_idx:04d}.jpg")
            cv2.imwrite(output_path, vis_frame)
            visualized_frames.append(output_path)
        
        # Créer une vidéo de toutes les frames visualisées
        video_path = os.path.join(output_dir, "tracking_visualization.mp4")
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
        
        for frame_path in visualized_frames:
            frame = cv2.imread(frame_path)
            if frame is not None:
                video.write(frame)
        
        video.release()
        
        return visualized_frames

    def _create_detection_visualizations(self, detection_stats: Dict[str, Any], output_dir: str):
        """
        Create additional visualizations from detection statistics
        """
        stats_dir = os.path.join(output_dir, 'statistics')
        os.makedirs(stats_dir, exist_ok=True)
        
        try:
            # Distribution des classes
            if detection_stats['object_counts']:
                plt.figure(figsize=(12, 8))
                classes = list(detection_stats['object_counts'].keys())
                counts = list(detection_stats['object_counts'].values())
                
                # Trier par fréquence
                sorted_indices = np.argsort(counts)[::-1]
                classes = [classes[i] for i in sorted_indices]
                counts = [counts[i] for i in sorted_indices]
                
                # Limiter à 15 classes pour la lisibilité
                if len(classes) > 15:
                    classes = classes[:15]
                    counts = counts[:15]
                    plt.title('Top 15 Detected Objects')
                else:
                    plt.title('Object Distribution')
                
                plt.bar(classes, counts)
                plt.xticks(rotation=45, ha='right')
                plt.ylabel('Detection Count')
                plt.tight_layout()
                plt.savefig(os.path.join(stats_dir, 'object_distribution.png'))
                plt.close()
            
            # Taux de détection
            plt.figure(figsize=(10, 6))
            detection_rate = detection_stats['frames_with_detections'] / detection_stats['total_frames'] * 100
            plt.bar(['Frames with detections', 'Frames without detections'], 
                    [detection_rate, 100 - detection_rate])
            plt.ylabel('Percentage')
            plt.title('Detection Rate by Frame')
            plt.savefig(os.path.join(stats_dir, 'detection_rate.png'))
            plt.close()
            
            # Histogramme de confiance
            if detection_stats['detection_confidence']:
                plt.figure(figsize=(10, 6))
                plt.hist(detection_stats['detection_confidence'], bins=20, range=(0, 1))
                plt.xlabel('Confidence Level')
                plt.ylabel('Detection Count')
                plt.title('Confidence Distribution')
                plt.grid(True, alpha=0.3)
                plt.savefig(os.path.join(stats_dir, 'confidence_histogram.png'))
                plt.close()
            
            # Résumé texte
            summary_path = os.path.join(output_dir, 'detection_summary.txt')
            with open(summary_path, 'w') as f:
                f.write("YOLO Detection Summary\n")
                f.write("========================\n\n")
                f.write(f"Total frames: {detection_stats['total_frames']}\n")
                f.write(f"Frames with detections: {detection_stats['frames_with_detections']} ")
                f.write(f"({detection_stats['frames_with_detections']/detection_stats['total_frames']*100:.1f}%)\n")
                f.write(f"Total detections: {detection_stats['total_detections']}\n")
                
                if detection_stats['detection_confidence']:
                    f.write(f"Average confidence: {detection_stats['avg_confidence']:.2f}\n")
                    f.write(f"Min/max confidence: {detection_stats['min_confidence']:.2f} / {detection_stats['max_confidence']:.2f}\n")
                
                f.write("\nObject counts:\n")
                for obj, count in sorted(detection_stats['object_counts'].items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{obj}: {count}\n")
        
        except Exception as e:
            print(f"[WARNING] Error creating visualizations: {e}")

    def calculate_similarity(self, 
                            track_hist, 
                            current_hist, 
                            track_pos, 
                            current_pos, 
                            track_size, 
                            current_size):
        """
        Calcule une similarité multicritère entre deux détections
        
        Args:
            track_hist: Histogramme de la track existante
            current_hist: Histogramme de la détection courante
            track_pos: Position [x, y] du centre de la track
            current_pos: Position [x, y] du centre de la détection
            track_size: Taille [w, h] de la track
            current_size: Taille [w, h] de la détection
            
        Returns:
            float: Score de similarité entre 0 et 1
        """
        # Similarité de couleur (comme avant mais normalisée)
        color_sim = cv2.compareHist(track_hist, current_hist, cv2.HISTCMP_CORREL)
        color_sim = (color_sim + 1) / 2  # Normalisation entre 0 et 1
        
        # Similarité de position (distance inversée)
        dist = np.sqrt((track_pos[0] - current_pos[0])**2 + (track_pos[1] - current_pos[1])**2)
        pos_sim = 1 / (1 + 0.005 * dist)  # Paramètre ajustable
        
        # Similarité de taille
        width_ratio = min(track_size[0]/current_size[0], current_size[0]/track_size[0])
        height_ratio = min(track_size[1]/current_size[1], current_size[1]/track_size[1])
        size_sim = (width_ratio + height_ratio) / 2
        
        # Combinaison pondérée
        final_sim = 0.6 * color_sim + 0.3 * pos_sim + 0.1 * size_sim
        
        return final_sim


    def predict_new_positions(self):
        """
        Prédit les nouvelles positions des tracks basées sur leur vitesse
        """
        for track_id, data in self.predicted_positions.items():
            x, y, vx, vy = data
            # Prédiction simple basée sur la vitesse
            self.predicted_positions[track_id] = [x + vx, y + vy, vx, vy]

    def update_velocity(self, track_id, new_pos, last_pos, alpha=0.7):
        """
        Met à jour la prédiction de vitesse pour une track
        
        Args:
            track_id: ID de la track
            new_pos: Nouvelle position [x, y]
            last_pos: Dernière position connue [x, y]
            alpha: Facteur de lissage pour la mise à jour
        """
        if track_id in self.predicted_positions:
            _, _, vx_old, vy_old = self.predicted_positions[track_id]
            vx_new = new_pos[0] - last_pos[0]
            vy_new = new_pos[1] - last_pos[1]
            
            # Lissage exponentiel
            vx = alpha * vx_new + (1 - alpha) * vx_old
            vy = alpha * vy_new + (1 - alpha) * vy_old
            
            self.predicted_positions[track_id] = [new_pos[0], new_pos[1], vx, vy]
        else:
            # Première détection, pas de vélocité
            self.predicted_positions[track_id] = [new_pos[0], new_pos[1], 0, 0]

    def update_track_velocity(self, track_id, new_center, frame_idx):
        """
        Met à jour la vélocité d'une track en utilisant la position actuelle
        
        Args:
            track_id: ID de la track à mettre à jour
            new_center: Nouvelle position du centre [x, y]
            frame_idx: Index de la frame courante
        """
        if track_id not in self.track_history:
            # Initialiser une nouvelle track
            self.track_history[track_id] = {
                'centers': [new_center],
                'frames': [frame_idx],
                'velocity': [0, 0],
                'last_seen': frame_idx
            }
            return
        
        track = self.track_history[track_id]
        
        # S'il y a des positions précédentes
        if 'centers' in track and len(track['centers']) > 0:
            last_center = track['centers'][-1]
            last_frame = track['frames'][-1]
            
            # Calculer le nombre de frames depuis la dernière mise à jour
            frames_delta = frame_idx - last_frame
            
            if frames_delta > 0:
                # Calculer la vélocité (déplacement par frame)
                vx = (new_center[0] - last_center[0]) / frames_delta
                vy = (new_center[1] - last_center[1]) / frames_delta
                
                # Si on a déjà une vélocité, faire une moyenne pondérée
                if 'velocity' in track:
                    old_vx, old_vy = track['velocity']
                    # Donner plus de poids aux mouvements récents
                    alpha = 0.7
                    vx = alpha * vx + (1 - alpha) * old_vx
                    vy = alpha * vy + (1 - alpha) * old_vy
                
                # Mettre à jour la vélocité
                track['velocity'] = [vx, vy]
        
        # Ajouter la nouvelle position
        track['centers'].append(new_center)
        track['frames'].append(frame_idx)
        track['last_seen'] = frame_idx


    def detect_and_track_person(self, frames, output_dir=None):
        """
        Version améliorée qui détecte les personnes frame par frame avec suivi plus robuste
        """
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        person_tracks = {}
        next_track_id = 0
        
        # Paramètres de suivi
        max_dist_threshold = 80  # Augmenté pour être moins strict
        max_frames_missing = self.max_frames_between_match
        active_tracks = {}  # {track_id: {'last_seen': frame_index, 'last_bbox': bbox, 'last_center': center, 'hist': hist}}
        
        # Prédire les nouvelles positions
        self.predict_new_positions()
        
        for frame_idx, frame in enumerate(frames):
            # Montrer la progression
            if frame_idx % 10 == 0:
                print(f"Analyse de la frame {frame_idx+1}/{len(frames)}")
            
            # Détecter avec YOLO
            try:
                results = self.yolo_model(frame, conf=self.confidence_threshold)[0]
                current_detections = []
                
                # Collecter toutes les détections de personnes dans cette frame
                for box in results.boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    # Vérifier si c'est une personne avec confiance suffisante
                    if cls == 0 and conf >= self.confidence_threshold:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                        center = [(x1 + x2) // 2, (y1 + y2) // 2]
                        size = [x2 - x1, y2 - y1]
                        
                        # Extraire l'histogramme de couleur
                        person_roi = frame[y1:y2, x1:x2]
                        hsv_roi = cv2.cvtColor(person_roi, cv2.COLOR_BGR2HSV)
                        h_hist = cv2.calcHist([hsv_roi], [0], None, [32], [0, 180])
                        s_hist = cv2.calcHist([hsv_roi], [1], None, [32], [0, 256])
                        color_hist = np.concatenate([
                            cv2.normalize(h_hist, h_hist).flatten(),
                            cv2.normalize(s_hist, s_hist).flatten()
                        ])
                        
                        current_detections.append({
                            'bbox': [x1, y1, x2, y2],
                            'center': center,
                            'size': size,
                            'hist': color_hist,
                            'conf': conf,
                            'matched': False
                        })
                
                # Pour chaque détection, essayer de l'associer à un track existant
                for det in current_detections:
                    best_track_id = None
                    best_similarity = 0.0
                    
                    # Chercher le track actif le plus similaire
                    for track_id, track_info in active_tracks.items():
                        # Obtenir les données de la dernière position connue
                        last_center = track_info['last_center']
                        last_hist = track_info['hist']
                        last_size = [
                            track_info['last_bbox'][2] - track_info['last_bbox'][0],
                            track_info['last_bbox'][3] - track_info['last_bbox'][1]
                        ]
                        
                        # Utiliser la fonction de similarité multicritère
                        similarity = self.calculate_similarity(
                            last_hist, 
                            det['hist'], 
                            last_center, 
                            det['center'], 
                            last_size, 
                            det['size']
                        )
                        
                        # Vérifier si c'est une meilleure correspondance
                        if similarity > best_similarity and similarity >= self.color_similarity_threshold:
                            best_similarity = similarity
                            best_track_id = track_id
                    
                    # Si on a trouvé une correspondance
                    if best_track_id is not None:
                        # Mettre à jour le track
                        last_center = active_tracks[best_track_id]['last_center']
                        active_tracks[best_track_id]['last_seen'] = frame_idx
                        active_tracks[best_track_id]['last_bbox'] = det['bbox']
                        active_tracks[best_track_id]['last_center'] = det['center']
                        active_tracks[best_track_id]['hist'] = det['hist']
                        
                        # Mettre à jour la prédiction de mouvement
                        self.update_velocity(best_track_id, det['center'], last_center)
                        
                        # Ajouter cette détection au track
                        if best_track_id not in person_tracks:
                            person_tracks[best_track_id] = {
                                'frames': [frame_idx],
                                'bboxes': [det['bbox']],
                                'centers': [det['center']],
                                'first_seen': frame_idx,
                                'last_seen': frame_idx
                            }
                        else:
                            person_tracks[best_track_id]['frames'].append(frame_idx)
                            person_tracks[best_track_id]['bboxes'].append(det['bbox'])
                            person_tracks[best_track_id]['centers'].append(det['center'])
                            person_tracks[best_track_id]['last_seen'] = frame_idx
                        
                        det['matched'] = True
                        
                    # Si on n'a pas trouvé de correspondance, créer un nouveau track
                    else:
                        new_track_id = next_track_id
                        next_track_id += 1
                        
                        # Créer un nouveau track
                        person_tracks[new_track_id] = {
                            'frames': [frame_idx],
                            'bboxes': [det['bbox']],
                            'centers': [det['center']],
                            'first_seen': frame_idx,
                            'last_seen': frame_idx
                        }
                        
                        # Ajouter aux tracks actifs
                        active_tracks[new_track_id] = {
                            'last_seen': frame_idx,
                            'last_bbox': det['bbox'],
                            'last_center': det['center'],
                            'hist': det['hist']
                        }
                        
                        # Initialiser la prédiction de mouvement
                        self.update_velocity(new_track_id, det['center'], det['center'])
                        
                        # Générer une couleur unique pour ce track
                        if new_track_id not in self.track_colors:
                            # Utiliser HSV pour des couleurs plus distinctes
                            hue = (new_track_id * 43) % 360  # 43 est premier avec 360
                            sat = 255
                            val = 255
                            color = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
                            self.track_colors[new_track_id] = tuple(map(int, color[0][0]))
                        
                        det['matched'] = True
                
                # Prédire de nouvelles positions pour les tracks non détectées
                self.predict_new_positions()
                
                # Supprimer les tracks inactifs depuis trop longtemps
                inactive_tracks = []
                for track_id, track_info in active_tracks.items():
                    if frame_idx - track_info['last_seen'] > max_frames_missing:
                        inactive_tracks.append(track_id)
                
                for track_id in inactive_tracks:
                    del active_tracks[track_id]
                    if track_id in self.predicted_positions:
                        del self.predicted_positions[track_id]
                
                # Le reste du code pour la visualisation reste similaire
                # Sauvegarder la visualisation si demandé
                if output_dir:
                    vis_frame = frame.copy()
                    
                    # Dessiner toutes les détections avec leur ID
                    for track_id, track_info in active_tracks.items():
                        if track_info['last_seen'] == frame_idx:  # Détection dans la frame courante
                            bbox = track_info['last_bbox']
                            x1, y1, x2, y2 = bbox
                            color = self.track_colors[track_id]
                            
                            # Dessiner le rectangle et l'ID
                            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                            label = f"ID {track_id}"
                            
                            # Fond pour le texte
                            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
                            cv2.rectangle(vis_frame, 
                                        (x1, y1 - text_size[1] - 10), 
                                        (x1 + text_size[0] + 10, y1), 
                                        color, -1)
                            
                            cv2.putText(vis_frame, label, (x1 + 5, y1 - 5), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                        elif frame_idx - track_info['last_seen'] <= max_frames_missing:
                            # Afficher les trackers prédits mais non détectés
                            if track_id in self.predicted_positions:
                                x, y, _, _ = self.predicted_positions[track_id]
                                x, y = int(x), int(y)
                                color = self.track_colors[track_id]
                                cv2.circle(vis_frame, (x, y), 5, color, -1)
                                cv2.putText(vis_frame, f"ID {track_id} (predicted)", 
                                          (x + 5, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.4, color, 1)
                    
                    # Ajouter des informations sur la frame
                    cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Nombre de tracks actifs
                    active_in_frame = sum(1 for info in active_tracks.values() if info['last_seen'] == frame_idx)
                    cv2.putText(vis_frame, f"Tracks actifs: {active_in_frame}", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    
                    # Sauvegarder la frame
                    vis_dir = os.path.join(output_dir, 'tracking')
                    os.makedirs(vis_dir, exist_ok=True)
                    cv2.imwrite(os.path.join(vis_dir, f'frame_{frame_idx:04d}.jpg'), vis_frame)
            
            except Exception as e:
                print(f"[WARNING] Erreur de détection sur frame {frame_idx}: {e}")
        
        # Filtrer les tracks trop courts (probablement des faux positifs)
        min_track_length = 3
        filtered_tracks = {}
        for track_id, track in person_tracks.items():
            if len(track['frames']) >= min_track_length:
                filtered_tracks[track_id] = track
        
        results = {
            'person_tracks': filtered_tracks,
            'total_tracks': len(filtered_tracks),
            'total_frames': len(frames)
        }
        
        print(f"Tracking terminé: {len(filtered_tracks)} tracks trouvés")
        
        return results
    
    def smooth_track_trajectories(self, person_tracks):
        """Lisse les trajectoires pour assurer un mouvement cohérent"""
        for track_id, track_data in person_tracks.items():
            if len(track_data['centers']) < 3:
                continue  # Pas assez de points pour le lissage
                
            # Convertir les centres en array numpy pour faciliter le traitement
            centers = np.array(track_data['centers'])
            frames = np.array(track_data['frames'])
            
            # Vérifier s'il y a des discontinuités importantes dans les frames
            frame_diffs = np.diff(frames)
            if np.max(frame_diffs) > 5:  # Discontinuité détectée
                # Traiter chaque segment continu séparément
                segments = []
                segment_start = 0
                
                for i, diff in enumerate(frame_diffs):
                    if diff > 5:
                        segments.append((segment_start, i + 1))
                        segment_start = i + 1
                segments.append((segment_start, len(frames)))
                
                # Lisser chaque segment
                smoothed_centers = centers.copy()
                for start, end in segments:
                    if end - start >= 3:  # Au moins 3 points pour lisser
                        segment_centers = centers[start:end]
                        # Filtre médian pour réduire les outliers
                        smoothed_segment = np.zeros_like(segment_centers)
                        for i in range(segment_centers.shape[1]):
                            smoothed_segment[:, i] = scipy.signal.medfilt(segment_centers[:, i], 3)
                        # Puis filtre de Savitzky-Golay pour lisser
                        window = min(5, end - start - 2)
                        window = window if window % 2 == 1 else window + 1  # S'assurer que window est impair
                        if window >= 3:
                            for i in range(segment_centers.shape[1]):
                                smoothed_segment[:, i] = scipy.signal.savgol_filter(
                                    smoothed_segment[:, i], window, 2)
                        smoothed_centers[start:end] = smoothed_segment
            else:
                # Pas de discontinuité, lissage sur l'ensemble
                smoothed_centers = centers.copy()
                # Filtre médian pour réduire les outliers
                for i in range(centers.shape[1]):
                    smoothed_centers[:, i] = scipy.signal.medfilt(centers[:, i], 3)
                # Puis filtre de Savitzky-Golay pour lisser
                window = min(7, len(centers) - 2)
                window = window if window % 2 == 1 else window - 1  # S'assurer que window est impair
                if window >= 3:
                    for i in range(centers.shape[1]):
                        smoothed_centers[:, i] = scipy.signal.savgol_filter(
                            smoothed_centers[:, i], window, 2)
            
            # Mettre à jour les centres lissés
            track_data['original_centers'] = track_data['centers'].copy()  # Sauvegarder les originaux
            track_data['centers'] = smoothed_centers.tolist()
            
            # Ajuster les bounding boxes pour qu'elles suivent les centres lissés
            for i, (old_center, new_center) in enumerate(zip(centers, smoothed_centers)):
                dx = new_center[0] - old_center[0]
                dy = new_center[1] - old_center[1]
                
                # Déplacer la bbox
                bbox = track_data['bboxes'][i]
                track_data['bboxes'][i] = [
                    bbox[0] + dx, bbox[1] + dy,
                    bbox[2] + dx, bbox[3] + dy
                ]
        
        return person_tracks

    def merge_similar_tracks(self, person_tracks, similarity_threshold=0.7):
        """
        Fusionne les tracks qui semblent appartenir à la même personne
        
        Args:
            person_tracks: Dictionnaire des tracks détectées
            similarity_threshold: Seuil pour considérer que deux tracks sont similaires
            
        Returns:
            Dictionnaire des tracks fusionnées
        """
        if not person_tracks:
            return {}
        
        # Trier les tracks par ordre chronologique de première apparition
        sorted_tracks = sorted(
            person_tracks.items(), 
            key=lambda x: x[1]['first_seen']
        )
        
        merged_tracks = {}
        track_mapping = {}  # Pour suivre à quelle track fusionnée appartient chaque track originale
        
        # Commencer avec la première track
        first_id, first_track = sorted_tracks[0]
        merged_tracks[first_id] = first_track.copy()
        track_mapping[first_id] = first_id
        
        # Pour chaque track restante
        for track_id, track_data in sorted_tracks[1:]:
            merged = False
            
            # Chercher une track similaire parmi les tracks déjà traitées
            for merged_id in merged_tracks.keys():
                # Vérifier si les tracks se chevauchent temporellement
                # Si oui, elles ne peuvent pas être fusionnées (ce sont des personnes différentes)
                if self._tracks_overlap(track_data, merged_tracks[merged_id]):
                    continue
                
                # Calculer la similarité entre les tracks
                similarity = self._calculate_track_similarity(track_data, merged_tracks[merged_id])
                
                if similarity >= similarity_threshold:
                    # Fusionner les tracks
                    self._merge_track_data(merged_tracks[merged_id], track_data)
                    track_mapping[track_id] = merged_id
                    merged = True
                    break
            
            # Si aucune fusion n'a été faite, ajouter comme nouvelle track
            if not merged:
                merged_tracks[track_id] = track_data.copy()
                track_mapping[track_id] = track_id
        
        print(f"Fusion des tracks: {len(person_tracks)} -> {len(merged_tracks)}")
        return merged_tracks, track_mapping

    def _tracks_overlap(self, track1, track2):
        """Vérifie si deux tracks se chevauchent temporellement"""
        # Extraire les frame indices
        frames1 = set(track1['frames'])
        frames2 = set(track2['frames'])
        
        # Vérifier l'intersection
        return len(frames1.intersection(frames2)) > 0

    def _calculate_track_similarity(self, track1, track2):
        """Calcule la similarité entre deux tracks"""
        # Facteurs de similarité
        
        # 1. Apparence (couleur)
        color_sim = 0.5  # Valeur par défaut
        
        # Si les tracks ont des boîtes, calculer la similarité de couleur
        if 'color_histograms' in track1 and 'color_histograms' in track2:
            # Moyenner les histogrammes de chaque track
            hist1 = np.mean(track1['color_histograms'], axis=0)
            hist2 = np.mean(track2['color_histograms'], axis=0)
            
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            color_sim = (correlation + 1) / 2
        
        # 2. Proximité spatiale (entre la dernière position de track1 et la première de track2 ou vice versa)
        position_sim = 0.5  # Valeur par défaut
        
        if track1['last_seen'] < track2['first_seen']:
            # track1 se termine avant track2
            if 'centers' in track1 and 'centers' in track2:
                last_pos_track1 = track1['centers'][-1]
                first_pos_track2 = track2['centers'][0]
                
                distance = np.sqrt((last_pos_track1[0] - first_pos_track2[0])**2 + 
                                (last_pos_track1[1] - first_pos_track2[1])**2)
                
                # Convertir en similarité (inversement proportionnelle à la distance)
                position_sim = 1.0 / (1.0 + distance / 100.0)
        else:
            # track2 se termine avant track1
            if 'centers' in track1 and 'centers' in track2:
                first_pos_track1 = track1['centers'][0]
                last_pos_track2 = track2['centers'][-1]
                
                distance = np.sqrt((first_pos_track1[0] - last_pos_track2[0])**2 + 
                                (first_pos_track1[1] - last_pos_track2[1])**2)
                
                position_sim = 1.0 / (1.0 + distance / 100.0)
        
        # 3. Proximité temporelle (l'écart de temps entre les tracks)
        time_gap = 0
        if track1['last_seen'] < track2['first_seen']:
            time_gap = track2['first_seen'] - track1['last_seen']
        else:
            time_gap = track1['first_seen'] - track2['last_seen']
        
        # Plus l'écart est petit, plus la similarité est grande
        temporal_sim = 1.0 / (1.0 + 0.1 * time_gap)
        
        # Similarité globale (moyenne pondérée)
        final_similarity = 0.4 * color_sim + 0.4 * position_sim + 0.2 * temporal_sim
        
        return final_similarity

    def _merge_track_data(self, target_track, source_track):
        """Fusionne les données de source_track dans target_track"""
        # Pour chaque attribut de liste
        for key in ['frames', 'bboxes', 'centers', 'confidences']:
            if key in source_track and key in target_track:
                target_track[key].extend(source_track[key])
        
        # Mettre à jour first_seen et last_seen
        if 'first_seen' in source_track and source_track['first_seen'] < target_track['first_seen']:
            target_track['first_seen'] = source_track['first_seen']
        
        if 'last_seen' in source_track and source_track['last_seen'] > target_track['last_seen']:
            target_track['last_seen'] = source_track['last_seen']
        
        # Trier les listes par frame_idx
        if 'frames' in target_track:
            # Créer des paires (frame_idx, élément) pour chaque attribut
            sorted_data = list(zip(target_track['frames'], 
                                target_track['bboxes'], 
                                target_track['centers']))
            
            # Trier par frame_idx
            sorted_data.sort(key=lambda x: x[0])
            
            # Décompresser les données triées
            frames, bboxes, centers = zip(*sorted_data)
            
            # Mettre à jour les attributs
            target_track['frames'] = list(frames)
            target_track['bboxes'] = list(bboxes)
            target_track['centers'] = list(centers)
            
            # Mettre à jour les confidences si disponibles
            if 'confidences' in target_track and len(target_track['confidences']) == len(sorted_data):
                sorted_conf = [(f, c) for f, c in zip(target_track['frames'], target_track['confidences'])]
                sorted_conf.sort(key=lambda x: x[0])
                target_track['confidences'] = [c for f, c in sorted_conf]


    def _create_tracking_visualizations(self, 
                                        person_tracks: Dict[int, Dict[str, Any]], 
                                        total_frames: int,
                                        output_dir: str):
        """
        Create additional visualizations for tracking
        
        Args:
            person_tracks: Tracking data
            total_frames: Total number of frames
            output_dir: Output directory
        """
        vis_dir = os.path.join(output_dir, 'tracking_analysis')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            # 1. Track duration
            plt.figure(figsize=(12, 8))
            track_ids = list(person_tracks.keys())
            durations = [track['last_seen'] - track['first_seen'] + 1 for track in person_tracks.values()]
            sorted_indices = np.argsort(durations)[::-1]
            track_ids = [track_ids[i] for i in sorted_indices]
            durations = [durations[i] for i in sorted_indices]
            
            plt.bar(range(len(track_ids)), durations)
            plt.xlabel('Track ID')
            plt.ylabel('Duration (frames)')
            plt.title('Person Track Durations')
            plt.xticks(range(len(track_ids)), track_ids, rotation=45)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'track_durations.png'))
            plt.close()
            
            # 2. Track timeline
            plt.figure(figsize=(12, 8))
            cmap = plt.cm.get_cmap('tab20', len(person_tracks))
            
            for i, (track_id, track) in enumerate(person_tracks.items()):
                first = track['first_seen']
                last = track['last_seen']
                plt.plot([first, last], [i, i], linewidth=10, 
                         color=cmap(i % cmap.N), label=f"ID {track_id}")
                
                frames = track['frames']
                plt.scatter(frames, [i] * len(frames), color=cmap(i % cmap.N), alpha=0.5)
            
            plt.yticks(range(len(person_tracks)), [f"ID {tid}" for tid in track_ids])
            plt.xlabel('Frame')
            plt.title('Person Presence Timeline')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'track_timeline.png'))
            plt.close()
            
            # 3. People per frame
            people_per_frame = np.zeros(total_frames, dtype=int)
            for track in person_tracks.values():
                for frame_idx in track['frames']:
                    if frame_idx < total_frames:
                        people_per_frame[frame_idx] += 1
            
            plt.figure(figsize=(12, 6))
            plt.plot(range(total_frames), people_per_frame)
            plt.xlabel('Frame')
            plt.ylabel('Number of People')
            plt.title('People Detected per Frame')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'people_per_frame.png'))
            plt.close()
            
            # 4. Position heatmap
            all_centers = []
            for track in person_tracks.values():
                all_centers.extend(track['centers'])
            
            if all_centers:
                centers = np.array(all_centers)
                hist, xedges, yedges = np.histogram2d(centers[:, 0], centers[:, 1], bins=[50, 50])
                
                plt.figure(figsize=(10, 8))
                plt.imshow(hist.T, origin='lower', aspect='auto', 
                           extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], 
                           cmap='hot')
                plt.colorbar(label='Frequency')
                plt.xlabel('X Position')
                plt.ylabel('Y Position')
                plt.title('Person Position Heatmap')
                plt.tight_layout()
                plt.savefig(os.path.join(vis_dir, 'position_heatmap.png'))
                plt.close()
            
            # Rapport texte
            with open(os.path.join(vis_dir, 'tracking_report.txt'), 'w') as f:
                f.write("Person Tracking Analysis Report\n")
                f.write("=======================================\n\n")
                f.write(f"Total frames analyzed: {total_frames}\n")
                f.write(f"Total people tracked: {len(person_tracks)}\n\n")
                
                f.write("Individual track summary:\n")
                for track_id, track in sorted(person_tracks.items(), 
                                              key=lambda x: x[1]['last_seen'] - x[1]['first_seen'] + 1, 
                                              reverse=True):
                    duration = track['last_seen'] - track['first_seen'] + 1
                    f.write(f"\nTrack ID {track_id}:\n")
                    f.write(f"  First appearance: frame {track['first_seen']}\n")
                    f.write(f"  Last appearance: frame {track['last_seen']}\n")
                    f.write(f"  Duration: {duration} frames ({duration / total_frames * 100:.1f}% of video)\n")
                    f.write(f"  Detections: {len(track['frames'])} frames "
                            f"({len(track['frames']) / duration * 100:.1f}% continuity)\n")
        
        except Exception as e:
            print(f"[WARNING] Error creating tracking visualizations: {e}")

    def simple_tracking_and_reordering(self, frames, output_dir=None):
        """
        Méthode de tracking basée principalement sur la similarité de couleur
        """
        if not frames:
            return frames
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        tracks = {}
        
        def calculate_color_histogram(bbox, frame):
            x1, y1, x2, y2 = map(int, bbox)
            box_region = frame[y1:y2, x1:x2]
            hsv_region = cv2.cvtColor(box_region, cv2.COLOR_BGR2HSV)
            h_hist = cv2.calcHist([hsv_region], [0], None, [180], [0, 180])
            s_hist = cv2.calcHist([hsv_region], [1], None, [256], [0, 256])
            h_hist = cv2.normalize(h_hist, h_hist).flatten()
            s_hist = cv2.normalize(s_hist, s_hist).flatten()
            return np.concatenate([h_hist, s_hist])
        
        def color_similarity(hist1, hist2):
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
            return (correlation + 1) / 2
        
        color_similarity_threshold = self.color_similarity_threshold
        max_frames_between_match = self.max_frames_between_match
        
        for frame_index, frame in enumerate(frames):
            results = self.yolo_model(frame, conf=self.confidence_threshold)[0]
            vis_frame = frame.copy() if output_dir else None
            
            for box in results.boxes:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                
                if cls == 0 and conf >= self.confidence_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bbox = (x1, y1, x2, y2)
                    color_hist = calculate_color_histogram(bbox, frame)
                    
                    best_track_id = None
                    best_similarity = 0
                    
                    for track_id, track_info in tracks.items():
                        if frame_index - track_info['last_seen'] > max_frames_between_match:
                            continue
                        
                        sim = color_similarity(track_info['color_hist'], color_hist)
                        if sim > best_similarity and sim >= color_similarity_threshold:
                            best_similarity = sim
                            best_track_id = track_id
                    
                    if best_track_id is None:
                        best_track_id = len(tracks)
                        tracks[best_track_id] = {
                            'frames': [frame_index],
                            'boxes': [bbox],
                            'color_hist': color_hist,
                            'first_seen': frame_index,
                            'last_seen': frame_index,
                            'total_similarity': 1.0
                        }
                    else:
                        tracks[best_track_id]['frames'].append(frame_index)
                        tracks[best_track_id]['boxes'].append(bbox)
                        tracks[best_track_id]['last_seen'] = frame_index
                        old_hist = tracks[best_track_id]['color_hist']
                        tracks[best_track_id]['color_hist'] = (
                            old_hist * (len(tracks[best_track_id]['frames']) - 1) + color_hist
                        ) / len(tracks[best_track_id]['frames'])
                    
                    if vis_frame is not None:
                        color_draw = (0, int(255 * best_similarity), 0)
                        cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color_draw, 2)
                        label = f"ID:{best_track_id} Sim:{best_similarity:.2f}"
                        cv2.putText(vis_frame, label, (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_draw, 2)
            
            if vis_frame is not None and output_dir:
                out_path = os.path.join(output_dir, f'color_tracking_{frame_index:04d}.jpg')
                cv2.imwrite(out_path, vis_frame)
        
        if not tracks:
            print("[WARNING] Aucune trajectoire détectée")
            return frames
        
        best_track_id = max(tracks.keys(), key=lambda k: len(tracks[k]['frames']))
        print(f"Réordonnancement basé sur la trajectoire ID {best_track_id} "
              f"(présente dans {len(tracks[best_track_id]['frames'])} frames)")
        
        frame_order = tracks[best_track_id]['frames']
        already_included = set(frame_order)
        for i in range(len(frames)):
            if i not in already_included:
                frame_order.append(i)
        
        reordered_frames = [frames[i] for i in frame_order]
        
        if output_dir:
            plt.figure(figsize=(12, 6))
            plt.plot(range(len(frame_order)), frame_order, 'b-')
            plt.title(f'Réordonnancement basé sur les couleurs (ID {best_track_id})')
            plt.xlabel('Nouvelle position')
            plt.ylabel('Position originale')
            plt.grid(True)
            plt.savefig(os.path.join(output_dir, 'color_reordering_plot.png'))
            plt.close()
        
        return reordered_frames

    def advanced_object_tracking(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Advanced object tracking method
        """
        tracks = {}
        
        for frame_index, frame in enumerate(frames):
            results = self.yolo_model(frame, conf=0.5)[0]
            for detection in results.boxes:
                class_id = int(detection.cls[0])
                confidence = float(detection.conf[0])
                bbox = detection.xyxy[0].tolist()
                
                if class_id == 0:  # Person
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    color_hist = self._calculate_color_histogram(frame, bbox)
                    
                    best_match = None
                    best_similarity = 0
                    for track_id, track_info in tracks.items():
                        last_bbox = track_info['bboxes'][-1]
                        last_center_x = (last_bbox[0] + last_bbox[2]) / 2
                        last_center_y = (last_bbox[1] + last_bbox[3]) / 2
                        distance = np.sqrt((center_x - last_center_x)**2 + (center_y - last_center_y)**2)
                        
                        size_similarity = min(
                            abs(bbox[2]-bbox[0]) / (abs(last_bbox[2]-last_bbox[0]) + 1e-5),
                            abs(bbox[3]-bbox[1]) / (abs(last_bbox[3]-last_bbox[1]) + 1e-5)
                        )
                        
                        color_similarity = cv2.compareHist(track_info['color_hist'], color_hist, cv2.HISTCMP_CORREL)
                        
                        similarity = (
                            0.4 / (1 + distance) + 
                            0.3 * size_similarity +
                            0.3 * (color_similarity + 1) / 2
                        )
                        
                        if similarity > best_similarity and similarity > 0.7:
                            best_match = track_id
                            best_similarity = similarity
                    
                    if best_match is None:
                        new_track_id = max(tracks.keys(), default=-1) + 1
                        tracks[new_track_id] = {
                            'frames': [frame_index],
                            'bboxes': [bbox],
                            'centers': [(center_x, center_y)],
                            'confidence': [confidence],
                            'color_hist': color_hist
                        }
                    else:
                        tracks[best_match]['frames'].append(frame_index)
                        tracks[best_match]['bboxes'].append(bbox)
                        tracks[best_match]['centers'].append((center_x, center_y))
                        tracks[best_match]['confidence'].append(confidence)
                        old_hist = tracks[best_match]['color_hist']
                        n = len(tracks[best_match]['frames'])
                        tracks[best_match]['color_hist'] = (old_hist * (n - 1) + color_hist) / n
        
        return tracks

    def _calculate_color_histogram(self, frame: np.ndarray, bbox: List[float]) -> np.ndarray:
        """
        Calculate a robust color histogram for an image region
        """
        x1, y1, x2, y2 = map(int, bbox)
        region = frame[y1:y2, x1:x2]
        hsv_region = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
        h_hist = cv2.calcHist([hsv_region], [0], None, [180], [0, 180])
        s_hist = cv2.calcHist([hsv_region], [1], None, [256], [0, 256])
        h_hist = cv2.normalize(h_hist, h_hist).flatten()
        s_hist = cv2.normalize(s_hist, s_hist).flatten()
        return np.concatenate([h_hist, s_hist])

    def reconstruct_video_with_tracking(self, 
                                        original_frames: List[np.ndarray], 
                                        tracked_objects: Dict[str, Any]) -> List[np.ndarray]:
        """
        Reconstruire la vidéo en se basant sur les tracks d'objets
        """
        if not tracked_objects:
            return original_frames
        
        main_track = max(tracked_objects.values(), key=lambda t: len(t['frames']))
        
        reordered_frames = []
        included_frames = set()
        
        for frame_idx in main_track['frames']:
            reordered_frames.append(original_frames[frame_idx])
            included_frames.add(frame_idx)
        
        for i, f in enumerate(original_frames):
            if i not in included_frames:
                reordered_frames.append(f)
        
        return reordered_frames

    def reorder_frames_by_tracking(self, 
                               tracking_results: Dict[str, Any],
                               frames: List[np.ndarray],
                               output_dir: Optional[str] = None) -> List[np.ndarray]:
        """
        Advanced frame reordering using sophisticated tracking techniques
        
        Args:
            tracking_results: Dictionary containing person tracking information
            frames: List of original frames 
            output_dir: Directory to save visualization and analysis
            
        Returns:
            List of reordered frames
        """
        if not tracking_results or 'person_tracks' not in tracking_results:
            print("[WARNING] No tracking results available for reordering")
            return frames

        person_tracks = tracking_results['person_tracks']
        if not person_tracks:
            print("[WARNING] No person tracks for reordering")
            return frames

        # Enhanced track selection and scoring mechanism
        def score_track(track):
            """Compute a comprehensive score for track quality"""
            if 'frames' not in track or len(track['frames']) == 0:
                return 0
            
            # Track length
            track_length = len(track['frames'])
            
            # Duration consistency
            frame_diffs = np.diff(track['frames'])
            duration_consistency = 1.0 / (np.std(frame_diffs) + 1)
            
            # Movement coherence 
            if 'centers' in track and len(track['centers']) > 1:
                centers = np.array(track['centers'])
                movement_vectors = np.diff(centers, axis=0)
                movement_coherence = np.mean(np.linalg.norm(movement_vectors, axis=1))
            else:
                movement_coherence = 0
            
            # Combine scoring factors
            score = (
                track_length * 0.4 +  # Longer tracks are preferred
                duration_consistency * 30 +  # Consistent frame progression
                movement_coherence * 10  # Smooth movement
            )
            
            return score

        # Find the best track for reordering
        best_track_id = max(person_tracks.keys(), key=lambda k: score_track(person_tracks[k]))
        best_track = person_tracks[best_track_id]
        
        print(f"Reordering based on Track ID {best_track_id}")
        
        # Extract frame indices from the best track
        track_frames = best_track['frames']
        
        # Create a comprehensive frame ordering strategy
        frame_order_map = {}
        
        # First, add frames from the main track
        for pos, original_idx in enumerate(track_frames):
            frame_order_map[original_idx] = pos
        
        # Add remaining frames systematically
        missing_frames = [i for i in range(len(frames)) if i not in frame_order_map]
        
        # Different strategies for adding missing frames
        strategies = [
            # 1. Closest time proximity to track
            lambda: sorted(missing_frames, 
                        key=lambda x: min(abs(x - track_frame) for track_frame in track_frames)),
            
            # 2. Random ordering of remaining frames
            lambda: np.random.permutation(missing_frames).tolist(),
            
            # 3. Original sequence order
            lambda: sorted(missing_frames)
        ]
        
        for strategy in strategies:
            if not frame_order_map:
                break
            
            remaining_frames = strategy()
            
            # Add missing frames to the end of the existing order
            current_max_pos = max(frame_order_map.values())
            for frame in remaining_frames:
                if frame not in frame_order_map:
                    current_max_pos += 1
                    frame_order_map[frame] = current_max_pos
        
        # Create the final reordered sequence
        reordered_indices = sorted(frame_order_map.keys(), key=frame_order_map.get)
        reordered_frames = [frames[idx] for idx in reordered_indices]
        
        # Optional: Visualization and analysis
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
            # Frame order visualization
            plt.figure(figsize=(15, 6))
            plt.title(f'Frame Reordering (Track ID {best_track_id})')
            plt.scatter(range(len(reordered_indices)), reordered_indices, alpha=0.7)
            plt.xlabel('New Position')
            plt.ylabel('Original Frame Index')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'frame_reordering.png'))
            plt.close()
            
            # Save reordering details
            with open(os.path.join(output_dir, 'reordering_details.json'), 'w') as f:
                json.dump({
                    'track_id': best_track_id,
                    'original_order': list(range(len(frames))),
                    'reordered_indices': reordered_indices
                }, f, indent=2)
        
        return reordered_frames

    def _visualize_reordering(self, frame_order, frames, output_dir, track_id):
        """
        Visualize the frame reordering
        """
        try:
            plt.figure(figsize=(12, 6))
            plt.scatter(range(len(frame_order)), frame_order)
            plt.plot(range(len(frame_order)), frame_order, 'r-', alpha=0.5)
            plt.xlabel('New position')
            plt.ylabel('Original position')
            plt.title(f'Frame reordering map based on track ID {track_id}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'frame_reordering_map.png'))
            plt.close()
            
            reordering_data = {
                'method': 'tracking',
                'track_id': str(track_id),
                'original_order': list(range(len(frames))),
                'reordered': [int(idx) for idx in frame_order if idx < len(frames)]
            }
            
            with open(os.path.join(output_dir, 'tracking_reordering_data.json'), 'w') as f:
                json.dump(reordering_data, f, indent=2)
        
        except Exception as e:
            print(f"[WARNING] Error visualizing reordering: {e}")


    def _visualize_advanced_reordering(self, 
                                  original_frames,
                                  reordered_frames,
                                  track_frames,
                                  reordering_indices,
                                  best_track,
                                  output_dir):
        """Visualisation avancée du processus de réordonnancement"""
        try:
            # 1. Graphique du réordonnancement
            plt.figure(figsize=(12, 8))
            plt.scatter(range(len(reordering_indices)), reordering_indices, alpha=0.7)
            plt.plot(range(len(reordering_indices)), reordering_indices, 'r-', alpha=0.3)
            
            # Mettre en évidence les frames du track principal
            track_indices = [i for i, idx in enumerate(reordering_indices) if idx in track_frames]
            track_original = [reordering_indices[i] for i in track_indices]
            plt.scatter(track_indices, track_original, color='green', s=100, alpha=0.7, 
                    label='Main Track Frames')
            
            plt.xlabel('New Position')
            plt.ylabel('Original Position')
            plt.title('Frame Reordering Map')
            plt.grid(True, alpha=0.3)
            plt.legend()
            plt.savefig(os.path.join(output_dir, 'reordering_map.png'))
            plt.close()
            
            # 2. Visualisation de la trajectoire du track principal
            plt.figure(figsize=(12, 8))
            centers = np.array(best_track['centers'])
            plt.plot(centers[:, 0], centers[:, 1], 'b-', alpha=0.5)
            plt.scatter(centers[:, 0], centers[:, 1], c=best_track['frames'], cmap='viridis')
            
            # Ajouter des flèches pour montrer la direction
            for i in range(0, len(centers)-1, max(1, len(centers)//20)):  # Afficher ~20 flèches
                plt.arrow(centers[i, 0], centers[i, 1], 
                        centers[i+1, 0] - centers[i, 0], centers[i+1, 1] - centers[i, 1],
                        head_width=10, head_length=10, fc='red', ec='red', alpha=0.7)
            
            plt.colorbar(label='Frame Index')
            plt.title('Main Track Trajectory')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'trajectory_map.png'))
            plt.close()
            
            # 3. Créer une vidéo de comparaison
            height, width = original_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            video_path = os.path.join(output_dir, 'reordering_comparison.mp4')
            video = cv2.VideoWriter(video_path, fourcc, 10, (width*2, height))
            
            for i in range(min(len(original_frames), len(reordered_frames))):
                # Frame originale avec son index
                orig_frame = original_frames[i].copy()
                cv2.putText(orig_frame, f"Original: {i}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Frame réordonnée avec son index original
                reord_frame = reordered_frames[i].copy()
                original_idx = reordering_indices[i]
                cv2.putText(reord_frame, f"Reordered (from {original_idx})", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                # Combiner côte à côte
                combined = np.hstack([orig_frame, reord_frame])
                video.write(combined)
            
            video.release()
            
            # 4. Export des données de réordonnancement
            reordering_data = {
                'track_id': str(best_track.get('id', 'unknown')),
                'track_frames': track_frames,
                'reordering_indices': reordering_indices,
                'original_indices': list(range(len(reordering_indices))),
                'monotonic': all(track_frames[i] <= track_frames[i+1] for i in range(len(track_frames)-1)),
                'track_length': len(track_frames)
            }
            
            with open(os.path.join(output_dir, 'reordering_data.json'), 'w') as f:
                json.dump(reordering_data, f, indent=2)
            
        except Exception as e:
            print(f"[WARNING] Error creating reordering visualizations: {e}")
            import traceback
            traceback.print_exc()

    def reorder_frames_by_tracking(self, frames):
        """
        Méthode de réordonnancement basée sur le tracking avec détection de direction
        """
        # Détecter et tracker les personnes
        tracking_results = self.detect_and_track_person(frames)
        person_tracks = tracking_results.get('person_tracks', {})
        
        if not person_tracks:
            return frames
        
        # Sélectionner la track principale
        best_track_id = max(person_tracks.keys(), key=lambda k: len(person_tracks[k]['frames']))
        best_track = person_tracks[best_track_id]
        
        # Détecter la direction de la séquence
        from frame_reorderer import AdvancedFrameReorderer
        sequence_direction = AdvancedFrameReorderer.detect_frame_sequence_direction(best_track)
        
        # Obtenir l'ordre des frames pour cette track
        track_frames = best_track['frames']
        
        # Réordonner les frames
        reordered_frames = [frames[idx] for idx in track_frames]
        
        # Inverser si nécessaire
        if sequence_direction == 'reversed':
            reordered_frames.reverse()
        
        return reordered_frames

    def reconstruct_tracking_video(self, 
                                   original_frames: List[np.ndarray], 
                                   reordered_frames: List[np.ndarray], 
                                   output_path: str,
                                   fps: float = 30.0):
        """
        Reconstruct a video with tracking visualization
        
        Args:
            original_frames: Original frames
            reordered_frames: Reordered frames
            output_path: Output path for the video
            fps: Frames per second
        """
        if not original_frames or not reordered_frames:
            print("[WARNING] No frames for tracking video")
            return
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Détecter et tracker dans les frames originales
        tracking_results = self.detect_and_track_person(original_frames)
        person_tracks = tracking_results['person_tracks']
        
        height, width = original_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
        
        print(f"Creating tracking video: {output_path}")
        
        for i in range(min(len(original_frames), len(reordered_frames))):
            orig_frame = original_frames[i].copy()
            reord_frame = reordered_frames[i].copy()
            
            for track_id, track in person_tracks.items():
                if i in track['frames']:
                    idx = track['frames'].index(i)
                    if idx < len(track['bboxes']):
                        bbox = track['bboxes'][idx]
                        x1, y1, x2, y2 = bbox
                        color = self.track_colors.get(track_id, (0, 255, 0))
                        cv2.rectangle(orig_frame, (x1, y1), (x2, y2), color, 2)
                        cv2.putText(orig_frame, f"ID: {track_id}", (x1, y1-10), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                        
                        centers = track['centers']
                        start_idx = max(0, idx - 20)
                        if idx > 0:
                            for j in range(start_idx, idx):
                                if j + 1 <= idx:
                                    cv2.line(orig_frame, 
                                             tuple(centers[j]), 
                                             tuple(centers[j+1]), 
                                             color, 2)
            
            cv2.putText(orig_frame, f"Original Frame #{i}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(reord_frame, f"Reordered Frame", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            combined_frame = np.hstack([orig_frame, reord_frame])
            writer.write(combined_frame)
        
        writer.release()
        print(f"Tracking video created: {output_path}")