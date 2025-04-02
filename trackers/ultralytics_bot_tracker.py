import cv2
import numpy as np
import os
from ultralytics import YOLO
from typing import List, Dict, Any, Optional

class UltralyticsTracker:
    """
    Tracker utilisant les capacités de tracking intégrées à Ultralytics YOLO
    """
    def __init__(self, 
                 yolo_model_path: str = 'yolov8m.pt',
                 confidence_threshold: float = 0.6,
                 tracker_type: str = "botsort",
                 visualization_enabled: bool = True):
        """
        Initialiser le tracker Ultralytics
        
        Args:
            yolo_model_path: Chemin vers le modèle YOLO
            confidence_threshold: Seuil de confiance pour la détection
            tracker_type: Type de tracker ('botsort', 'bytetrack', etc.)
            visualization_enabled: Activer la visualisation automatique
        """
        # Charger le modèle YOLO
        self.yolo_model = YOLO(yolo_model_path)
        print(f"Modèle YOLO chargé: {yolo_model_path}")
        
        # Configuration
        self.confidence_threshold = confidence_threshold
        self.tracker_type = tracker_type
        self.visualization_enabled = visualization_enabled
        
        # Dictionnaire pour couleurs des IDs de tracking
        self.track_colors = {}
        
        print(f"UltralyticsTracker initialisé avec {tracker_type}")
    
    def detect_and_track_person(self, frames: List[np.ndarray], output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Détecter et suivre les personnes dans une séquence de frames
        
        Args:
            frames: Liste des frames (images numpy)
            output_dir: Répertoire de sortie pour les visualisations
            
        Returns:
            Dictionnaire contenant les tracks de personnes détectées
        """
        print(f"Analyse de {len(frames)} frames avec {self.tracker_type}...")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        person_tracks = {}
        
        for frame_idx, frame in enumerate(frames):
            # Montrer la progression
            if frame_idx % 10 == 0:
                print(f"Analyse de la frame {frame_idx+1}/{len(frames)}")
            
            # Détecter et tracker avec YOLO
            results = self.yolo_model.track(frame, 
                                           conf=self.confidence_threshold, 
                                           persist=True)  # Maintenir les IDs entre frames
            
            # Extraire les résultats de tracking
            if hasattr(results[0], 'boxes') and hasattr(results[0].boxes, 'id'):
                boxes = results[0].boxes
                
                for i, box in enumerate(boxes):
                    # Vérifier si c'est une personne (classe 0)
                    if int(box.cls[0]) == 0:
                        # Vérifier si l'ID de tracking existe
                        if box.id is not None:
                            track_id = int(box.id[0])
                            bbox = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                            x1, y1, x2, y2 = bbox
                            center = [(x1 + x2) / 2, (y1 + y2) / 2]
                            
                            # Ajouter ou mettre à jour la track
                            if track_id not in person_tracks:
                                person_tracks[track_id] = {
                                    'frames': [frame_idx],
                                    'bboxes': [bbox],
                                    'centers': [center],
                                    'first_seen': frame_idx,
                                    'last_seen': frame_idx
                                }
                                
                                # Générer une couleur unique pour ce track
                                if track_id not in self.track_colors:
                                    hue = (track_id * 43) % 360  # 43 est premier avec 360
                                    sat = 255
                                    val = 255
                                    color = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
                                    self.track_colors[track_id] = tuple(map(int, color[0][0]))
                            else:
                                person_tracks[track_id]['frames'].append(frame_idx)
                                person_tracks[track_id]['bboxes'].append(bbox)
                                person_tracks[track_id]['centers'].append(center)
                                person_tracks[track_id]['last_seen'] = frame_idx
            
            # Visualisations si demandé
            if self.visualization_enabled and output_dir:
                # Créer une visualisation en utilisant le résultat de YOLO
                plotted_frame = results[0].plot()  # Utilise la visualisation intégrée d'Ultralytics
                
                # Ajouter des informations supplémentaires
                cv2.putText(plotted_frame, f"Frame: {frame_idx}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Sauvegarder la frame
                vis_dir = os.path.join(output_dir, 'tracking')
                os.makedirs(vis_dir, exist_ok=True)
                cv2.imwrite(os.path.join(vis_dir, f'frame_{frame_idx:04d}.jpg'), plotted_frame)
        
        # Créer des statistiques de tracking
        stats = {
            'total_tracks': len(person_tracks),
            'average_track_length': 0,
            'longest_track_length': 0,
        }
        
        if person_tracks:
            track_lengths = [len(track['frames']) for track in person_tracks.values()]
            stats['average_track_length'] = sum(track_lengths) / len(track_lengths)
            stats['longest_track_length'] = max(track_lengths)
        
        print(f"Tracking terminé: {len(person_tracks)} tracks trouvés")
        
        return {
            'person_tracks': person_tracks,
            'stats': stats,
            'total_tracks': len(person_tracks),
            'total_frames': len(frames)
        }
    
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
                        if track_id not in self.track_colors:
                            hue = (track_id * 43) % 360
                            sat = 255
                            val = 255
                            color = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
                            self.track_colors[track_id] = tuple(map(int, color[0][0]))
                        
                        color = self.track_colors[track_id]
                        
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
        if visualized_frames:
            try:
                video_path = os.path.join(output_dir, "tracking_visualization.mp4")
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
                
                for frame_path in visualized_frames:
                    frame = cv2.imread(frame_path)
                    if frame is not None:
                        video.write(frame)
                
                video.release()
                print(f"Vidéo de visualisation créée: {video_path}")
            except Exception as e:
                print(f"Impossible de créer la vidéo de visualisation: {e}")
        
        return visualized_frames