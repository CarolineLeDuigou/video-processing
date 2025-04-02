# AutoTrackReorderUtils.py
from typing import Dict, List, Any
import numpy as np

class AutoTrackReorderUtils:
    @staticmethod
    def is_track_reversed(track_data: Dict[str, Any], area_threshold_ratio: float = -0.3) -> bool:
        """
        Détecte si une track semble inversée en analysant l'évolution de la taille de la bounding box.
        """
        areas = []
        for bbox in track_data['bboxes']:
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)
            areas.append(area)
        if len(areas) < 5:
            return False
        delta = areas[-1] - areas[0]
        ratio = delta / areas[0] if areas[0] != 0 else 0
        return ratio < area_threshold_ratio

    @staticmethod
    def auto_reverse_all_tracks(person_tracks: Dict[int, Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        """
        Applique automatiquement un reverse() sur les tracks qui semblent inversées.
        """
        corrected_tracks = {}
        for track_id, track_data in person_tracks.items():
            if AutoTrackReorderUtils.is_track_reversed(track_data):
                print(f"[AUTO-REVERSE] Track {track_id} inversée automatiquement")
                track_data['frames'].reverse()
                track_data['bboxes'].reverse()
            corrected_tracks[track_id] = track_data
        return corrected_tracks

    @staticmethod
    def _calculate_score(bbox: List[int], frame_idx: int, expected_idx: int, alpha: float = 0.5, beta: float = 0.2) -> float:
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        center_y = (y1 + y2) / 2.0
        temporal_penalty = -beta * abs(frame_idx - expected_idx)
        return area + alpha * center_y + temporal_penalty

    @staticmethod
    def global_frame_reordering_with_smoothing(person_tracks: Dict[int, Dict[str, Any]], alpha: float = 0.5) -> List[int]:
        """
        Fusionne globalement les frames de toutes les tracks avec :
         1. Calcul d'un score combinant la surface et la position verticale.
         2. Tri initial par ordre croissant de score.
         3. Lissage local pour corriger les inversions ponctuelles.
        """
        combined_frames_info = []
        for track_id, track_data in person_tracks.items():
            for idx, (frame_idx, bbox) in enumerate(zip(track_data['frames'], track_data['bboxes'])):
                score = AutoTrackReorderUtils._calculate_score(bbox, frame_idx=frame_idx, expected_idx=idx, alpha=alpha, beta=0.2)
                combined_frames_info.append({
                    'frame_idx': frame_idx,
                    'score': score,
                    'track_id': track_id
                })
        # Éliminer les doublons en gardant la meilleure score pour une même frame
        unique_frames = {}
        for info in combined_frames_info:
            idx = info['frame_idx']
            if idx not in unique_frames or info['score'] > unique_frames[idx]['score']:
                unique_frames[idx] = info

        ordered = sorted(unique_frames.values(), key=lambda x: x['score'])
        # Appliquer le lissage local pour corriger les inversions
        corrected_indices = AutoTrackReorderUtils._smooth_local_inversions(ordered, window_size=7, threshold_ratio=0.3)
        return corrected_indices

    @staticmethod
    def _smooth_local_inversions(ordered_frames: List[Dict[str, Any]], window_size: int = 7, jump_threshold: float = 0.15) -> List[int]:
        """
        Corrige les inversions locales brutales en analysant les sauts de score (brusques diminutions).
        Si une frame a un score bien plus faible que ses voisines (saut brutal),
        on la déplace intelligemment dans une meilleure position.
        """
        corrected = ordered_frames.copy()
        changed = True

        def relative_drop(a, b):
            return (a - b) / max(a, 1e-6)  # pour éviter division par 0

        while changed:
            changed = False
            for i in range(1, len(corrected) - 1):
                prev = corrected[i - 1]['score']
                curr = corrected[i]['score']
                next = corrected[i + 1]['score']

                drop_prev = relative_drop(prev, curr)
                drop_next = relative_drop(next, curr)

                # Si le score actuel est très en-dessous des voisins → suspect
                if drop_prev > jump_threshold and drop_next > jump_threshold:
                    # Cherche un meilleur endroit dans une fenêtre autour
                    best_j = i
                    for j in range(max(0, i - window_size), min(len(corrected), i + window_size)):
                        if corrected[j]['score'] > curr:
                            best_j = j
                            break
                    if best_j != i:
                        corrected.insert(best_j, corrected.pop(i))
                        changed = True
                        break  # on recommence avec le nouveau ordre

        return [f['frame_idx'] for f in corrected]

    @staticmethod
    def global_frame_reordering_from_tracks(person_tracks: Dict[int, Dict[str, Any]]) -> List[int]:
        """
        Fusionne les frames de toutes les tracks en estimant un ordre temporel global.
        Corrige aussi les tracks inversées + détecte si la séquence complète est à l’envers.
        """
        # Étape 1 : auto-reverse des tracks si nécessaire
        corrected_tracks = AutoTrackReorderUtils.auto_reverse_all_tracks(person_tracks)

        # Étape 2 : fusion de toutes les frames avec un score
        combined_frames_info = []

        for track_id, track_data in corrected_tracks.items():
            for frame_idx, bbox in zip(track_data['frames'], track_data['bboxes']):
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                center_y = (y1 + y2) / 2
                score = area + center_y
                combined_frames_info.append({
                    'frame_idx': frame_idx,
                    'score': score,
                    'track_id': track_id
                })

        # Supprimer les doublons
        unique_frames = {}
        for info in combined_frames_info:
            idx = info['frame_idx']
            if idx not in unique_frames or info['score'] > unique_frames[idx]['score']:
                unique_frames[idx] = info

        # Étape 3 : tri croissant sur le score estimé
        ordered = sorted(unique_frames.values(), key=lambda x: x['score'])
        ordered_indices = [f['frame_idx'] for f in ordered]

        # Étape 4 : vérifier si la séquence est inversée
        if len(ordered) > 5:
            first_area = ordered[0]['score']
            last_area = ordered[-1]['score']
            ratio = (last_area - first_area) / max(first_area, 1e-5)
            if ratio < -0.2:
                print("[AUTO FIX] La séquence semble inversée globalement → on inverse l’ordre.")
                ordered_indices.reverse()

        return ordered_indices
    

    @staticmethod
    def _reorder_by_spatial_flow(ordered_frames: List[Dict[str, Any]]) -> List[int]:
        """
        Réordonne les frames pour assurer une continuité fluide dans le déplacement spatial.
        """
        centers = [
            ((f['bbox'][0] + f['bbox'][2]) / 2, (f['bbox'][1] + f['bbox'][3]) / 2)
            for f in ordered_frames
        ]
        frame_ids = [f['frame_idx'] for f in ordered_frames]

        # On applique un tri basé sur la progression cumulative du centre
        cumulative = [0]
        total = 0
        for i in range(1, len(centers)):
            dx = centers[i][0] - centers[i-1][0]
            dy = centers[i][1] - centers[i-1][1]
            total += np.sqrt(dx**2 + dy**2)
            cumulative.append(total)

        ordered_with_flow = sorted(zip(frame_ids, cumulative), key=lambda x: x[1])
        return [f[0] for f in ordered_with_flow]