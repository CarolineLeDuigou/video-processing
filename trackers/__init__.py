"""
Package de trackers pour le suivi d'objets dans des vidéos.

Ce module fournit différentes implémentations de trackers compatibles
avec l'interface commune BaseTracker.
"""

from .person_tracker import PersonTracker
from trackers.ultralytics_bot_tracker import UltralyticsTracker

# Essayer d'importer StrongTracker s'il est disponible
try:
    from trackers.person_tracker_strongsort import StrongPersonTracker
    __all__ = ['PersonTracker', 'UltralyticsTracker', 'StrongPersonTracker']
except ImportError:
    __all__ = ['PersonTracker', 'UltralyticsTracker']

def get_tracker(tracker_type, **kwargs):
    """
    Factory method pour obtenir une instance du tracker demandé.
    
    Args:
        tracker_type (str): Type de tracker ('deepsort', 'botsort', 'bytetrack', 'strongsort')
        **kwargs: Arguments additionnels à passer au constructeur du tracker
        
    Returns:
        Une instance du tracker demandé
    
    Raises:
        ValueError: Si le type de tracker n'est pas reconnu
    """
    tracker_map = {
        'deepsort': PersonTracker,
        'botsort': lambda **kw: UltralyticsTracker(tracker_type="botsort", **kw),
        'bytetrack': lambda **kw: UltralyticsTracker(tracker_type="bytetrack", **kw),
    }
    
    # Tenter d'ajouter StrongSORT s'il est disponible
    if 'StrongPersonTracker' in globals():
        tracker_map['strongsort'] = StrongPersonTracker
    
    if tracker_type not in tracker_map:
        available_types = ", ".join(tracker_map.keys())
        raise ValueError(f"Type de tracker '{tracker_type}' non reconnu. "
                         f"Types disponibles: {available_types}")
    
    # Créer et retourner l'instance du tracker
    tracker_class = tracker_map[tracker_type]
    return tracker_class(**kwargs)