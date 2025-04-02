import cv2
import numpy as np
import os
from ultralytics import YOLO
import supervision as sv
from typing import List, Dict, Any, Optional

class BotPersonTracker:
    def __init__(self, 
                 yolo_model_path: str = 'yolov8m.pt',
                 confidence_threshold: float = 0.6,
                 visualization_enabled: bool = True):
        
        self.yolo_model = YOLO(yolo_model_path)
        print(f"Modèle YOLO chargé: {yolo_model_path}")
        
        self.confidence_threshold = confidence_threshold
        self.visualization_enabled = visualization_enabled
        
        # Version corrigée - vérifiez la façon correcte d'initialiser ByteTrack
        # Dans les versions récentes de supervision, ByteTrack est une classe de tracker
        self.tracker = sv.ByteTrack()
        
        self.track_colors = {}
        self.coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack'
        ]
        
        print("BotPersonTracker initialisé avec succès")

    def detect_and_track_person(self, frames: List[np.ndarray], output_dir: Optional[str] = None) -> Dict[str, Any]:
        print(f"Analyse de {len(frames)} frames avec BoT-SORT...")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        person_tracks = {}
        
        # Créer un annotator pour les visualisations
        box_annotator = sv.BoxAnnotator()
        
        for frame_idx, frame in enumerate(frames):
            if frame_idx % 10 == 0:
                print(f"Analyse de la frame {frame_idx+1}/{len(frames)}")
            
            # Détecter avec YOLO
            results = self.yolo_model(frame, conf=self.confidence_threshold)[0]
            
            # Convertir les résultats YOLO au format supervision
            detections = sv.Detections.from_ultralytics(results)
            
            # Filtrer pour ne garder que les personnes (classe 0)
            if len(detections) > 0:
                mask = detections.class_id == 0
                detections = detections[mask]
            
            # CORRECTION: Utiliser la bonne méthode pour mettre à jour le tracker
            # Le tracker de supervision renvoie un objet Detections avec des tracker_id
            tracked_objects = self.tracker.track(detections)
            
            # Traiter les détections trackées
            for i in range(len(tracked_objects)):
                if tracked_objects.tracker_id is None or len(tracked_objects.tracker_id) <= i:
                    continue
                
                track_id = int(tracked_objects.tracker_id[i])
                bbox = tracked_objects.xyxy[i]
                x1, y1, x2, y2 = bbox
                
                # Calculer le centre
                center = [(x1 + x2) / 2, (y1 + y2) / 2]
                
                # Ajouter au dictionnaire des tracks
                if track_id not in person_tracks:
                    person_tracks[track_id] = {
                        'frames': [frame_idx],
                        'bboxes': [[float(x1), float(y1), float(x2), float(y2)]],
                        'centers': [center],
                        'first_seen': frame_idx,
                        'last_seen': frame_idx
                    }
                    
                    # Générer une couleur unique pour ce track_id
                    if track_id not in self.track_colors:
                        hue = (track_id * 43) % 360  # 43 est premier avec 360
                        sat = 255
                        val = 255
                        color = cv2.cvtColor(np.array([[[hue, sat, val]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)
                        self.track_colors[track_id] = tuple(map(int, color[0][0]))
                else:
                    person_tracks[track_id]['frames'].append(frame_idx)
                    person_tracks[track_id]['bboxes'].append([float(x1), float(y1), float(x2), float(y2)])
                    person_tracks[track_id]['centers'].append(center)
                    person_tracks[track_id]['last_seen'] = frame_idx
            
            # Visualisations si demandé
            if self.visualization_enabled and output_dir:
                vis_frame = frame.copy()
                
                # Dessiner les détections avec l'annotateur de supervision
                if tracked_objects is not None and len(tracked_objects) > 0:
                    # Créer les labels avec les IDs
                    labels = [f"ID: {id}" for id in tracked_objects.tracker_id] if tracked_objects.tracker_id is not None else []
                    
                    # Dessiner les bounding boxes
                    vis_frame = box_annotator.annotate(vis_frame, tracked_objects, labels=labels)
                
                # Ajouter des informations sur la frame
                cv2.putText(vis_frame, f"Frame: {frame_idx}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
                # Sauvegarder la frame
                vis_dir = os.path.join(output_dir, 'tracking')
                os.makedirs(vis_dir, exist_ok=True)
                cv2.imwrite(os.path.join(vis_dir, f'frame_{frame_idx:04d}.jpg'), vis_frame)
        
        print(f"Tracking terminé: {len(person_tracks)} tracks trouvés")
        
        return {
            'person_tracks': person_tracks,
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
        
        # Créer un annotateur supervision
        box_annotator = sv.BoxAnnotator()
        
        # Traiter chaque frame
        for frame_idx, frame in enumerate(frames):
            # Copier la frame pour la visualisation
            vis_frame = frame.copy()
            
            # Créer une liste de detections pour cette frame
            xyxy_list = []
            track_id_list = []
            confidence_list = []
            class_id_list = []
            
            # Vérifier quelles tracks sont présentes dans cette frame
            for track_id, track_data in person_tracks.items():
                if frame_idx in track_data['frames']:
                    # Obtenir l'index dans les données de tracking
                    track_idx = track_data['frames'].index(frame_idx)
                    
                    # Récupérer la bounding box
                    if track_idx < len(track_data['bboxes']):
                        bbox = track_data['bboxes'][track_idx]
                        xyxy_list.append(bbox)
                        track_id_list.append(track_id)
                        confidence_list.append(1.0)  # Valeur par défaut
                        class_id_list.append(0)  # Classe 0 = personne
            
            # Création d'un objet Detections de supervision
            if xyxy_list:
                detections = sv.Detections(
                    xyxy=np.array(xyxy_list),
                    confidence=np.array(confidence_list),
                    class_id=np.array(class_id_list),
                    tracker_id=np.array(track_id_list)
                )
                
                # Créer les labels
                labels = [f"ID: {id}" for id in track_id_list]
                
                # Dessiner les bounding boxes et labels
                vis_frame = box_annotator.annotate(vis_frame, detections)
                
                # Dessiner les labels manuellement
                for i, (x1, y1, x2, y2) in enumerate(xyxy_list):
                    track_id = track_id_list[i]
                    color = self.track_colors.get(track_id, (0, 255, 0))
                    
                    # Dessiner le texte
                    label = f"ID: {track_id}"
                    cv2.putText(vis_frame, label, (int(x1), int(y1) - 10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Ajouter l'information de frame
            cv2.putText(vis_frame, f"Frame: {frame_idx}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
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