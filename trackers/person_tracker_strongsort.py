import os
import cv2
import torch
import numpy as np
from ultralytics import YOLO
from typing import List, Dict, Any

# Essayer les différentes importations possibles de StrongSORT
try:
    # Essai 1: chemin original mentionné en commentaire
    from strong_sort_onnx.strong_sort import StrongSORT
    STRONG_SORT_SOURCE = "strong_sort_onnx"
except ImportError:
    try:
        # Essai 2: chemin modifié comme indiqué dans le code
        from trackers.strong_sort.strong_sort import StrongSORT
        STRONG_SORT_SOURCE = "trackers.strong_sort"
    except ImportError:
        try:
            # Essai 3: chemin relatif local
            import sys
            sys.path.append('Yolov5_StrongSORT_OSNet')
            from trackers.strong_sort.strong_sort import StrongSORT
            STRONG_SORT_SOURCE = "Yolov5_StrongSORT_OSNet.trackers.strong_sort"
        except ImportError:
            raise ImportError("Impossible de trouver le module StrongSORT. Veuillez vérifier l'installation et les chemins.")

class StrongPersonTracker:
    def __init__(self, yolo_model_path='yolov8m.pt', strongsort_model_path='models/osnet_x0_25_msmt17.onnx'):
        """
        Initialise le tracker de personnes basé sur StrongSORT
        
        Args:
            yolo_model_path: Chemin vers le modèle YOLO
            strongsort_model_path: Chemin vers le modèle OSNet pour StrongSORT
        """
        print(f"[INFO] Initialisation de StrongPersonTracker avec StrongSORT de {STRONG_SORT_SOURCE}")
        print(f"[INFO] Modèle YOLO: {yolo_model_path}")
        print(f"[INFO] Modèle OSNet: {strongsort_model_path}")
        
        self.yolo_model = YOLO(yolo_model_path)
        
        # Vérifier que le fichier du modèle existe
        if not os.path.exists(strongsort_model_path):
            # Chercher dans les emplacements alternatifs
            alt_paths = [
                'models/osnet_x0_25_msmt17.onnx',
                'Yolov5_StrongSORT_OSNet/models/osnet_x0_25_msmt17.onnx',
                'Yolov5_StrongSORT_OSNet/trackers/strong_sort/models/osnet_x0_25_msmt17.onnx'
            ]
            
            for alt_path in alt_paths:
                if os.path.exists(alt_path):
                    strongsort_model_path = alt_path
                    print(f"[INFO] Modèle trouvé dans un emplacement alternatif: {strongsort_model_path}")
                    break
        
        # Initialiser StrongSORT
        try:
            self.tracker = StrongSORT(
                model_path=strongsort_model_path,
                device='cuda' if torch.cuda.is_available() else 'cpu'
            )
            print(f"[INFO] StrongSORT initialisé avec succès sur {self.tracker.device}")
        except Exception as e:
            raise Exception(f"Erreur lors de l'initialisation de StrongSORT: {e}")

    def detect_and_track_person(self, frames: List[np.ndarray]) -> Dict[str, Any]:
        """
        Détecte et suit les personnes dans une séquence de frames
        
        Args:
            frames: Liste des frames (images OpenCV)
            
        Returns:
            Dictionnaire contenant les tracks de personnes détectées
        """
        person_tracks = {}

        for frame_idx, frame in enumerate(frames):
            results = self.yolo_model(frame)[0]
            boxes = results.boxes

            bboxes = []
            confidences = []

            for box in boxes:
                cls_id = int(box.cls[0])
                if cls_id == 0:  # classe 'person'
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    w, h = x2 - x1, y2 - y1
                    conf = float(box.conf[0])
                    bboxes.append([x1, y1, w, h])  # xywh format
                    confidences.append(conf)

            bboxes = np.array(bboxes)
            confidences = np.array(confidences)

            # Update du tracker
            if len(bboxes) > 0:
                outputs = self.tracker.update(bboxes, confidences, frame)

                for det in outputs:
                    x1, y1, x2, y2, track_id = map(int, det[:5])
                    if track_id not in person_tracks:
                        person_tracks[track_id] = {'frames': [], 'bboxes': [], 'centers': []}
                    person_tracks[track_id]['frames'].append(frame_idx)
                    person_tracks[track_id]['bboxes'].append([x1, y1, x2, y2])
                    person_tracks[track_id]['centers'].append([(x1 + x2) // 2, (y1 + y2) // 2])
            
            if frame_idx % 10 == 0:
                print(f"Traitement de la frame {frame_idx}/{len(frames)}")

        # Ajouter des métadonnées supplémentaires aux tracks
        for track_id, track_data in person_tracks.items():
            if track_data['frames']:
                track_data['first_seen'] = track_data['frames'][0]
                track_data['last_seen'] = track_data['frames'][-1]

        return {
            'person_tracks': person_tracks,
            'total_tracks': len(person_tracks)
        }

    def visualize_tracking_ids(self, frames: List[np.ndarray], tracking_results: Dict[str, Any], output_dir: str) -> List[str]:
        """
        Sauvegarde les frames avec les track IDs affichés
        
        Args:
            frames: Liste des frames
            tracking_results: Résultats du tracking
            output_dir: Répertoire de sortie
            
        Returns:
            Liste des chemins des frames visualisées
        """
        os.makedirs(output_dir, exist_ok=True)
        paths = []

        person_tracks = tracking_results.get('person_tracks', {})

        # Inverser : frame_id → liste d'objets trackés
        frame_map = {}
        for tid, data in person_tracks.items():
            for idx, bbox in zip(data['frames'], data['bboxes']):
                if idx < len(frames):  # Vérification pour éviter les erreurs
                    frame_map.setdefault(idx, []).append((tid, bbox))

        for idx, frame in enumerate(frames):
            if idx in frame_map:
                annotated = frame.copy()
                for tid, (x1, y1, x2, y2) in frame_map[idx]:
                    # Générer une couleur unique pour chaque track_id
                    color_h = (tid * 43) % 180  # 43 est premier avec 180
                    color = tuple(map(int, cv2.cvtColor(np.array([[[color_h, 255, 255]]], dtype=np.uint8), cv2.COLOR_HSV2BGR)[0, 0]))
                    
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, f"ID: {tid}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                
                # Ajouter un titre à la frame
                cv2.putText(annotated, f"Frame {idx}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                
                out_path = os.path.join(output_dir, f"frame_{idx:04d}.jpg")
                cv2.imwrite(out_path, annotated)
                paths.append(out_path)

        # Créer une vidéo des frames visualisées si possible
        if paths:
            try:
                video_path = os.path.join(output_dir, "tracking_visualization.mp4")
                height, width = frames[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                video = cv2.VideoWriter(video_path, fourcc, 10, (width, height))
                
                for path in paths:
                    frame = cv2.imread(path)
                    if frame is not None:
                        video.write(frame)
                
                video.release()
                print(f"[INFO] Vidéo de visualisation créée: {video_path}")
            except Exception as e:
                print(f"[WARNING] Impossible de créer la vidéo de visualisation: {e}")

        return paths

    def visualize_yolo_detections(self, frame_paths: List[str], output_dir: str,
                                 save_all_frames=True, export_csv=True) -> List[str]:
        """
        Visualise les détections YOLO sur chaque frame
        
        Args:
            frame_paths: Liste des chemins vers les images
            output_dir: Dossier de sortie
            save_all_frames: Sauvegarder toutes les frames
            export_csv: Exporter les détections en CSV
            
        Returns:
            Liste des chemins des visualisations
        """
        os.makedirs(output_dir, exist_ok=True)
        visualization_paths = []
        
        detection_stats = {
            'total_frames': len(frame_paths),
            'frames_with_detections': 0,
            'total_detections': 0,
            'object_counts': {}
        }
        
        if export_csv:
            import csv
            csv_path = os.path.join(output_dir, 'yolo_detections.csv')
            csv_file = open(csv_path, 'w', newline='')
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow([
                'frame_index', 'frame_path', 'class_id', 'class_name',
                'confidence', 'x1', 'y1', 'x2', 'y2', 'width', 'height'
            ])
        else:
            csv_writer = None
        
        coco_classes = [
            'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
            'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 
            'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack'
        ]
        
        print(f"[INFO] Analyse de {len(frame_paths)} frames avec YOLO...")
        
        for frame_index, frame_path in enumerate(frame_paths):
            if frame_index % 10 == 0:
                print(f"Traitement de la frame {frame_index}/{len(frame_paths)}")
            
            frame = cv2.imread(frame_path)
            if frame is None:
                continue
            
            # Exécuter les détections
            results = self.yolo_model(frame)[0]
            boxes = results.boxes
            
            # Traiter les détections
            has_detections = len(boxes) > 0
            if has_detections:
                detection_stats['frames_with_detections'] += 1
                
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    
                    class_name = coco_classes[cls] if cls < len(coco_classes) else f"class_{cls}"
                    
                    detection_stats['total_detections'] += 1
                    detection_stats['object_counts'][class_name] = detection_stats['object_counts'].get(class_name, 0) + 1
                    
                    # Dessiner la détection
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    color = (0, 255, 0)  # Vert
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if csv_writer:
                        width, height = x2 - x1, y2 - y1
                        csv_writer.writerow([
                            frame_index, os.path.basename(frame_path),
                            cls, class_name, conf,
                            x1, y1, x2, y2, width, height
                        ])
            
            # Sauvegarder la visualisation
            if save_all_frames or has_detections:
                output_path = os.path.join(output_dir, f'yolo_{frame_index:04d}.jpg')
                cv2.imwrite(output_path, frame)
                visualization_paths.append(output_path)
        
        if csv_writer:
            csv_file.close()
        
        # Sauvegarder les statistiques
        import json
        stats_path = os.path.join(output_dir, 'detection_statistics.json')
        with open(stats_path, 'w') as f:
            json.dump(detection_stats, f, indent=2)
        
        return visualization_paths