import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Dict, Any
import json
from datetime import datetime

class VideoUtils:
    """
    Utilitaires pour l'extraction et le traitement des frames vidéo
    """
    
    @staticmethod
    def extract_frames(video_path: str, 
                     output_dir: str = 'frames', 
                     max_frames: Optional[int] = None,
                     save_analysis: bool = True,
                     verbose: bool = True) -> List[str]:
        """
        Extraire les frames d'une vidéo avec diagnostic détaillé
        
        Args:
            video_path (str): Chemin vers la vidéo source
            output_dir (str): Répertoire de sortie des frames
            max_frames (Optional[int]): Nombre maximal de frames à extraire
            save_analysis (bool): Sauvegarder les analyses et visualisations
            verbose (bool): Afficher des informations détaillées
        
        Returns:
            List[str]: Chemins des frames extraites
        """
        # Créer le répertoire de sortie et ses sous-répertoires
        os.makedirs(output_dir, exist_ok=True)
        
        # Créer un sous-répertoire pour les analyses
        analysis_dir = os.path.join(output_dir, 'analysis')
        if save_analysis:
            os.makedirs(analysis_dir, exist_ok=True)
        
        # Diagnostic préliminaire du fichier vidéo
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Le fichier vidéo {video_path} n'existe pas.")
        
        # Taille et type de fichier
        file_size = os.path.getsize(video_path)
        file_extension = os.path.splitext(video_path)[1]
        
        if verbose:
            print(f"[DIAGNOSTIC] Taille du fichier : {file_size} octets")
            print(f"[DIAGNOSTIC] Extension du fichier : {file_extension}")
        
        # Tentative d'ouverture de la vidéo
        cap = cv2.VideoCapture(video_path)
        
        # Vérifier si la capture est réussie
        if not cap.isOpened():
            raise IOError(f"Impossible d'ouvrir la vidéo : {video_path}")
        
        # Informations sur la vidéo
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        if verbose:
            print(f"[DIAGNOSTIC] FPS : {fps}")
            print(f"[DIAGNOSTIC] Nombre total de frames : {frame_count}")
            print(f"[DIAGNOSTIC] Dimensions : {width}x{height}")
            print(f"[DIAGNOSTIC] Durée estimée : {duration:.2f} secondes")
        
        # Créer un dictionnaire pour stocker les métadonnées
        video_metadata = {
            "file_path": video_path,
            "file_size": file_size,
            "file_extension": file_extension,
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration,
            "extraction_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "max_frames_limit": max_frames
        }
        
        # Sauvegarder les métadonnées au format JSON
        if save_analysis:
            with open(os.path.join(analysis_dir, 'video_metadata.json'), 'w') as f:
                json.dump(video_metadata, f, indent=2)
        
        # Limiter le nombre de frames si spécifié
        if max_frames is not None:
            frame_count = min(frame_count, max_frames)
        
        # Arrays pour stocker les histogrammes et autres métriques
        if save_analysis:
            histograms = []
            brightness_values = []
            sharpness_values = []
        
        # Extraction des frames
        frames = []
        success = True
        frame_index = 0
        
        while success and (max_frames is None or frame_index < max_frames):
            success, frame = cap.read()
            
            if not success:
                break
            
            # Sauvegarder la frame
            frame_path = os.path.join(output_dir, f'frame_{frame_index:04d}.jpg')
            cv2.imwrite(frame_path, frame)
            frames.append(frame_path)
            
            # Collecter des données pour l'analyse
            if save_analysis and (frame_index % 5 == 0 or frame_count < 50):  # Analyser 1 frame sur 5 si nombreuses
                # Calculer l'histogramme de couleur
                hist_b = cv2.calcHist([frame], [0], None, [256], [0, 256])
                hist_g = cv2.calcHist([frame], [1], None, [256], [0, 256])
                hist_r = cv2.calcHist([frame], [2], None, [256], [0, 256])
                histograms.append((hist_b, hist_g, hist_r, frame_index))
                
                # Calculer la luminosité moyenne
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                brightness = np.mean(gray)
                brightness_values.append((frame_index, brightness))
                
                # Calculer la netteté (utilisant la variance du Laplacien)
                laplacian = cv2.Laplacian(gray, cv2.CV_64F)
                sharpness = np.var(laplacian)
                sharpness_values.append((frame_index, sharpness))
                
                # Sauvegarder une version miniature pour la vignette
                if frame_index % 20 == 0 or frame_count < 20:
                    thumbnail_dir = os.path.join(analysis_dir, 'thumbnails')
                    os.makedirs(thumbnail_dir, exist_ok=True)
                    thumbnail = cv2.resize(frame, (width//4, height//4))
                    cv2.imwrite(os.path.join(thumbnail_dir, f'thumbnail_{frame_index:04d}.jpg'), thumbnail)
            
            frame_index += 1
            
            # Afficher la progression
            if verbose and frame_index % 100 == 0:
                print(f"[PROGRESSION] Frames extraites : {frame_index}")
        
        cap.release()
        
        if verbose:
            print(f"[RÉSULTAT] Total frames extraites : {len(frames)}")
        
        if len(frames) == 0:
            raise ValueError("Aucune frame n'a pu être extraite. Vérifiez le fichier vidéo.")
        
        # Générer des visualisations
        if save_analysis:
            # Créer des répertoires pour les visualisations
            vis_dir = os.path.join(analysis_dir, 'visualizations')
            os.makedirs(vis_dir, exist_ok=True)
            
            # Générer une mosaïque d'aperçu
            VideoUtils.create_frame_montage(frames, os.path.join(vis_dir, 'frame_montage.jpg'), 
                                          max_frames=25, columns=5)
            
            try:
                # Visualisation des histogrammes
                if histograms:
                    plt.figure(figsize=(12, 8))
                    for i, (b, g, r, idx) in enumerate(histograms[:10]):  # Max 10 histos pour la lisibilité
                        plt.subplot(5, 2, i+1)
                        plt.plot(b, color='blue')
                        plt.plot(g, color='green')
                        plt.plot(r, color='red')
                        plt.title(f'Frame {idx}')
                        plt.xlim([0, 256])
                    plt.tight_layout()
                    plt.savefig(os.path.join(vis_dir, 'color_histograms.png'))
                    plt.close()
                
                # Visualisation de la luminosité
                if brightness_values:
                    plt.figure(figsize=(12, 6))
                    frame_indices, brightness = zip(*brightness_values)
                    plt.plot(frame_indices, brightness)
                    plt.title('Évolution de la luminosité')
                    plt.xlabel('Index de frame')
                    plt.ylabel('Luminosité moyenne')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(vis_dir, 'brightness_evolution.png'))
                    plt.close()
                
                # Visualisation de la netteté
                if sharpness_values:
                    plt.figure(figsize=(12, 6))
                    frame_indices, sharpness = zip(*sharpness_values)
                    plt.plot(frame_indices, sharpness)
                    plt.title('Évolution de la netteté')
                    plt.xlabel('Index de frame')
                    plt.ylabel('Netteté (variance du Laplacien)')
                    plt.grid(True, alpha=0.3)
                    plt.savefig(os.path.join(vis_dir, 'sharpness_evolution.png'))
                    plt.close()
            
            except Exception as e:
                print(f"[AVERTISSEMENT] Erreur lors de la création des visualisations: {e}")
            
            # Générer un rapport d'extraction
            report = {
                "video_metadata": video_metadata,
                "frames_extracted": len(frames),
                "frames_paths": frames,
                "extraction_success": True,
                "brightness_stats": {
                    "mean": np.mean([b for _, b in brightness_values]) if brightness_values else 0,
                    "min": np.min([b for _, b in brightness_values]) if brightness_values else 0,
                    "max": np.max([b for _, b in brightness_values]) if brightness_values else 0
                },
                "sharpness_stats": {
                    "mean": np.mean([s for _, s in sharpness_values]) if sharpness_values else 0,
                    "min": np.min([s for _, s in sharpness_values]) if sharpness_values else 0,
                    "max": np.max([s for _, s in sharpness_values]) if sharpness_values else 0
                }
            }
            
            # Sauvegarder le rapport
            with open(os.path.join(analysis_dir, 'extraction_report.json'), 'w') as f:
                json.dump(report, f, indent=2)
            
            # Créer un résumé textuel
            with open(os.path.join(analysis_dir, 'extraction_summary.txt'), 'w') as f:
                f.write("Résumé de l'extraction des frames\n")
                f.write("==============================\n\n")
                f.write(f"Vidéo source: {video_path}\n")
                f.write(f"Taille du fichier: {file_size} octets\n")
                f.write(f"Dimensions: {width}x{height}\n")
                f.write(f"FPS: {fps}\n")
                f.write(f"Nombre de frames annoncé: {frame_count}\n")
                f.write(f"Durée: {duration:.2f} secondes\n\n")
                f.write(f"Frames extraites: {len(frames)}\n")
                
                if brightness_values:
                    f.write("\nStatistiques de luminosité:\n")
                    f.write(f"  Moyenne: {report['brightness_stats']['mean']:.2f}\n")
                    f.write(f"  Min: {report['brightness_stats']['min']:.2f}\n")
                    f.write(f"  Max: {report['brightness_stats']['max']:.2f}\n")
                
                if sharpness_values:
                    f.write("\nStatistiques de netteté:\n")
                    f.write(f"  Moyenne: {report['sharpness_stats']['mean']:.2f}\n")
                    f.write(f"  Min: {report['sharpness_stats']['min']:.2f}\n")
                    f.write(f"  Max: {report['sharpness_stats']['max']:.2f}\n")
        
        return frames
    
    @staticmethod
    def create_frame_montage(frame_paths: List[str], 
                            output_path: str, 
                            max_frames: int = 16, 
                            columns: int = 4) -> bool:
        """
        Créer une mosaïque d'aperçu des frames
        
        Args:
            frame_paths: Liste des chemins des frames
            output_path: Chemin du fichier de sortie
            max_frames: Nombre maximal de frames à inclure
            columns: Nombre de colonnes dans la mosaïque
        
        Returns:
            bool: True si la mosaïque a été créée avec succès
        """
        if not frame_paths:
            return False
        
        # Limiter le nombre de frames
        if max_frames > 0:
            # Sélectionner les frames de manière équidistante
            if len(frame_paths) > max_frames:
                step = len(frame_paths) // max_frames
                selected_paths = [frame_paths[i] for i in range(0, len(frame_paths), step)][:max_frames]
            else:
                selected_paths = frame_paths[:max_frames]
        else:
            selected_paths = frame_paths
        
        # Déterminer le nombre de lignes
        rows = (len(selected_paths) + columns - 1) // columns
        
        # Lire la première image pour déterminer les dimensions
        sample = cv2.imread(selected_paths[0])
        if sample is None:
            print(f"[ERREUR] Impossible de lire l'image: {selected_paths[0]}")
            return False
        
        height, width = sample.shape[:2]
        
        # Définir la taille des miniatures
        thumb_width = 320
        thumb_height = int(height * thumb_width / width)
        
        # Créer une image vide pour la mosaïque
        montage = np.zeros((thumb_height * rows, thumb_width * columns, 3), dtype=np.uint8)
        
        # Remplir la mosaïque
        for i, path in enumerate(selected_paths):
            row = i // columns
            col = i % columns
            
            img = cv2.imread(path)
            if img is not None:
                # Redimensionner l'image
                resized = cv2.resize(img, (thumb_width, thumb_height))
                
                # Placer dans la mosaïque
                y_start = row * thumb_height
                x_start = col * thumb_width
                montage[y_start:y_start+thumb_height, x_start:x_start+thumb_width] = resized
                
                # Ajouter l'index
                cv2.putText(montage, f"{i}", (x_start + 5, y_start + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Sauvegarder la mosaïque
        cv2.imwrite(output_path, montage)
        
        return True
    
    @staticmethod
    def analyze_video_quality(frames: List[str], 
                             output_dir: str = 'quality_analysis') -> Dict[str, Any]:
        """
        Analyser la qualité des frames extraites
        
        Args:
            frames: Liste des chemins des frames
            output_dir: Répertoire de sortie pour les résultats
        
        Returns:
            Dict avec les métriques d'analyse
        """
        if not frames:
            return {'status': 'error', 'message': 'Aucune frame à analyser'}
        
        # Créer le répertoire de sortie
        os.makedirs(output_dir, exist_ok=True)
        
        # Métriques à calculer
        brightness = []
        contrast = []
        sharpness = []
        noise = []
        
        # Collecter frames avec problèmes potentiels
        dark_frames = []
        bright_frames = []
        blurry_frames = []
        noisy_frames = []
        
        # Analyse des frames
        print(f"Analyse de la qualité de {len(frames)} frames...")
        
        for i, frame_path in enumerate(frames):
            # Lire l'image
            img = cv2.imread(frame_path)
            if img is None:
                print(f"[AVERTISSEMENT] Impossible de lire la frame: {frame_path}")
                continue
            
            # Convertir en niveaux de gris
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Calculer la luminosité (moyenne des pixels)
            mean_brightness = np.mean(gray)
            brightness.append(mean_brightness)
            
            # Calculer le contraste (écart-type des pixels)
            std_dev = np.std(gray)
            contrast.append(std_dev)
            
            # Calculer la netteté (variance du Laplacien)
            laplacian = cv2.Laplacian(gray, cv2.CV_64F)
            sharpness_value = np.var(laplacian)
            sharpness.append(sharpness_value)
            
            # Calculer une estimation du bruit
            # Appliquer un filtre médian pour réduire le bruit
            denoised = cv2.medianBlur(gray, 5)
            noise_level = np.mean(np.abs(gray.astype(np.float32) - denoised.astype(np.float32)))
            noise.append(noise_level)
            
            # Identifier les problèmes potentiels
            if mean_brightness < 50:  # Frame sombre
                dark_frames.append((i, frame_path, mean_brightness))
            
            if mean_brightness > 200:  # Frame trop claire
                bright_frames.append((i, frame_path, mean_brightness))
            
            if sharpness_value < 50:  # Frame floue
                blurry_frames.append((i, frame_path, sharpness_value))
            
            if noise_level > 10:  # Frame bruitée
                noisy_frames.append((i, frame_path, noise_level))
            
            # Afficher la progression
            if i % 100 == 0:
                print(f"Analyse de la frame {i+1}/{len(frames)}")
        
        # Statistiques globales
        if not brightness:
            return {'status': 'error', 'message': 'Aucune donnée collectée'}
        
        avg_brightness = np.mean(brightness)
        avg_contrast = np.mean(contrast)
        avg_sharpness = np.mean(sharpness)
        avg_noise = np.mean(noise)
        
        # Créer des visualisations
        vis_dir = os.path.join(output_dir, 'visualizations')
        os.makedirs(vis_dir, exist_ok=True)
        
        try:
            # Évolution de la luminosité
            plt.figure(figsize=(12, 6))
            plt.plot(brightness)
            plt.axhline(y=avg_brightness, color='r', linestyle='--', label=f'Moyenne ({avg_brightness:.2f})')
            plt.title('Évolution de la luminosité')
            plt.xlabel('Index de frame')
            plt.ylabel('Luminosité moyenne')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'brightness_evolution.png'))
            plt.close()
            
            # Évolution de la netteté
            plt.figure(figsize=(12, 6))
            plt.plot(sharpness)
            plt.axhline(y=avg_sharpness, color='r', linestyle='--', label=f'Moyenne ({avg_sharpness:.2f})')
            plt.title('Évolution de la netteté')
            plt.xlabel('Index de frame')
            plt.ylabel('Netteté (variance du Laplacien)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'sharpness_evolution.png'))
            plt.close()
            
            # Évolution du contraste
            plt.figure(figsize=(12, 6))
            plt.plot(contrast)
            plt.axhline(y=avg_contrast, color='r', linestyle='--', label=f'Moyenne ({avg_contrast:.2f})')
            plt.title('Évolution du contraste')
            plt.xlabel('Index de frame')
            plt.ylabel('Contraste (écart-type)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'contrast_evolution.png'))
            plt.close()
            
            # Évolution du bruit
            plt.figure(figsize=(12, 6))
            plt.plot(noise)
            plt.axhline(y=avg_noise, color='r', linestyle='--', label=f'Moyenne ({avg_noise:.2f})')
            plt.title('Évolution du niveau de bruit')
            plt.xlabel('Index de frame')
            plt.ylabel('Niveau de bruit estimé')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'noise_evolution.png'))
            plt.close()
            
            # Scatter plot contraste vs luminosité
            plt.figure(figsize=(10, 8))
            plt.scatter(brightness, contrast, alpha=0.5)
            plt.xlabel('Luminosité')
            plt.ylabel('Contraste')
            plt.title('Relation Luminosité-Contraste')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'brightness_vs_contrast.png'))
            plt.close()
            
            # Scatter plot netteté vs bruit
            plt.figure(figsize=(10, 8))
            plt.scatter(sharpness, noise, alpha=0.5)
            plt.xlabel('Netteté')
            plt.ylabel('Bruit')
            plt.title('Relation Netteté-Bruit')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(vis_dir, 'sharpness_vs_noise.png'))
            plt.close()
        
        except Exception as e:
            print(f"[AVERTISSEMENT] Erreur lors de la création des visualisations: {e}")
        
        # Créer des mosaïques pour les frames problématiques
        issues_dir = os.path.join(output_dir, 'problematic_frames')
        os.makedirs(issues_dir, exist_ok=True)
        
        # Sauvegarder des frames d'exemple pour chaque problème
        def save_examples(frames_list, issue_type, max_examples=10):
            if not frames_list:
                return
            
            examples_dir = os.path.join(issues_dir, issue_type)
            os.makedirs(examples_dir, exist_ok=True)
            
            # Limiter le nombre d'exemples
            examples = frames_list[:max_examples]
            
            for idx, path, value in examples:
                img = cv2.imread(path)
                if img is not None:
                    # Ajouter une annotation
                    cv2.putText(img, f"{issue_type}: {value:.2f}", (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    
                    # Sauvegarder l'exemple
                    example_path = os.path.join(examples_dir, f'frame_{idx:04d}.jpg')
                    cv2.imwrite(example_path, img)
            
            # Créer une mosaïque
            example_paths = [os.path.join(examples_dir, f'frame_{ex[0]:04d}.jpg') for ex in examples]
            montage_path = os.path.join(examples_dir, f'{issue_type}_montage.jpg')
            VideoUtils.create_frame_montage(example_paths, montage_path)
        
        save_examples(dark_frames, 'dark')
        save_examples(bright_frames, 'bright')
        save_examples(blurry_frames, 'blurry')
        save_examples(noisy_frames, 'noisy')
        
        # Générer un rapport d'analyse
        analysis_results = {
            'status': 'success',
            'total_frames': len(frames),
            'brightness': {
                'mean': float(avg_brightness),
                'min': float(np.min(brightness)),
                'max': float(np.max(brightness)),
                'std': float(np.std(brightness))
            },
            'contrast': {
                'mean': float(avg_contrast),
                'min': float(np.min(contrast)),
                'max': float(np.max(contrast)),
                'std': float(np.std(contrast))
            },
            'sharpness': {
                'mean': float(avg_sharpness),
                'min': float(np.min(sharpness)),
                'max': float(np.max(sharpness)),
                'std': float(np.std(sharpness))
            },
            'noise': {
                'mean': float(avg_noise),
                'min': float(np.min(noise)),
                'max': float(np.max(noise)),
                'std': float(np.std(noise))
            },
            'potential_issues': {
                'dark_frames': len(dark_frames),
                'bright_frames': len(bright_frames),
                'blurry_frames': len(blurry_frames),
                'noisy_frames': len(noisy_frames)
            }
        }
        
        # Sauvegarder les résultats au format JSON
        with open(os.path.join(output_dir, 'quality_analysis.json'), 'w') as f:
            json.dump(analysis_results, f, indent=2)
        
        # Créer un rapport textuel
        with open(os.path.join(output_dir, 'quality_report.txt'), 'w') as f:
            f.write("Rapport d'analyse de qualité vidéo\n")
            f.write("================================\n\n")
            f.write(f"Nombre total de frames analysées: {len(frames)}\n\n")
            
            f.write("Statistiques de luminosité:\n")
            f.write(f"  Moyenne: {avg_brightness:.2f}\n")
            f.write(f"  Min: {np.min(brightness):.2f}\n")
            f.write(f"  Max: {np.max(brightness):.2f}\n")
            f.write(f"  Écart-type: {np.std(brightness):.2f}\n\n")
            
            f.write("Statistiques de contraste:\n")
            f.write(f"  Moyenne: {avg_contrast:.2f}\n")
            f.write(f"  Min: {np.min(contrast):.2f}\n")
            f.write(f"  Max: {np.max(contrast):.2f}\n")
            f.write(f"  Écart-type: {np.std(contrast):.2f}\n\n")
            
            f.write("Statistiques de netteté:\n")
            f.write(f"  Moyenne: {avg_sharpness:.2f}\n")
            f.write(f"  Min: {np.min(sharpness):.2f}\n")
            f.write(f"  Max: {np.max(sharpness):.2f}\n")
            f.write(f"  Écart-type: {np.std(sharpness):.2f}\n\n")
            
            f.write("Statistiques de bruit:\n")
            f.write(f"  Moyenne: {avg_noise:.2f}\n")
            f.write(f"  Min: {np.min(noise):.2f}\n")
            f.write(f"  Max: {np.max(noise):.2f}\n")
            f.write(f"  Écart-type: {np.std(noise):.2f}\n\n")
            
            f.write("Problèmes potentiels détectés:\n")
            f.write(f"  Frames sombres: {len(dark_frames)} ({len(dark_frames)/len(frames)*100:.1f}%)\n")
            f.write(f"  Frames trop claires: {len(bright_frames)} ({len(bright_frames)/len(frames)*100:.1f}%)\n")
            f.write(f"  Frames floues: {len(blurry_frames)} ({len(blurry_frames)/len(frames)*100:.1f}%)\n")
            f.write(f"  Frames bruitées: {len(noisy_frames)} ({len(noisy_frames)/len(frames)*100:.1f}%)\n")
        
        return analysis_results

# Exemple de test
def main():
    video_path = 'corrupted_video.mp4'
    try:
        # Extraire les frames avec analyse complète
        frames = VideoUtils.extract_frames(
            video_path, 
            output_dir='video_frames', 
            save_analysis=True, 
            verbose=True
        )
        
        print(f"Frames extraites avec succès : {len(frames)}")
        
        # Analyser la qualité des frames
        analysis = VideoUtils.analyze_video_quality(frames, output_dir='quality_analysis')
        
        print("\nRésumé de l'analyse de qualité:")
        print(f"Luminosité moyenne: {analysis['brightness']['mean']:.2f}")
        print(f"Netteté moyenne: {analysis['sharpness']['mean']:.2f}")
        print(f"Frames problématiques: {sum(analysis['potential_issues'].values())}")
    
    except Exception as e:
        print(f"Erreur lors de l'extraction : {e}")

if __name__ == "__main__":
    main()