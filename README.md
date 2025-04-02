# Video Processing Framework

Un framework Python avancé pour le traitement, l'analyse et la correction de vidéos corrompues ou désordonnées, avec un focus sur la réorganisation intelligente des frames et le suivi d'objets.


## 🌟 Caractéristiques principales

- **Réordonnancement intelligent des frames vidéo** - Utilise plusieurs stratégies pour reconstruire l'ordre correct des frames:
  - Tracking d'objets et personnes
  - Analyse de flux optique 
  - Méthodes basées sur les caractéristiques visuelles
  - Tri topologique et approche hybride

- **Détection robuste d'outliers** - Identifie les frames aberrantes ou corrompues via:
  - Analyse en Composantes Principales (PCA)
  - Méthode statistique (Z-score)
  - Isolation Forest
  - Clustering et détection d'anomalies

- **Tracking multi-objets performant** avec support pour:
  - DeepSORT
  - ByteTrack
  - BoT-SORT

- **Extraction avancée de caractéristiques** avec analyse de textures, couleurs et mouvements

- **Visualisations détaillées** pour l'analyse et le diagnostic

- **Architecture hautement configurable** s'adaptant à différents cas d'usage
  

## 🔧 Installation

### Prérequis
- Python 3.7 ou plus récent
- GPU recommandé pour le tracking d'objets (mais non obligatoire)

### Installation simple
```
git clone https://github.com/CarolineLeDuigou/video-processing.git
cd video-processing
pip install -r requirements.txt
```

### Installation avec environnement virtuel (recommandé)
```
git clone https://github.com/votre-username/video-processing.git
cd video-processing
python -m venv venv
source venv/bin/activate  # ou venv\Scripts\activate sur Windows
pip install -r requirements.txt
```

## 💻 Exemples d'utilisation en ligne de commande

### Traitement simple
python main.py --input corrupted_video.mp4 --output corrected_video.mp4

### Avec toutes les visualisations
python main.py --input corrupted_video.mp4 --visualize-all --analyze-objects

### Sélection d'un tracker et d'une méthode de réordonnement spécifiques
python main.py --input corrupted_video.mp4 --reordering fused --tracker deepsort --visualize-all 

