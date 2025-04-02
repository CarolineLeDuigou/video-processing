"""
Package de traitement vidéo pour réordonner et analyser des vidéos.

Ce package fournit des outils pour détecter et corriger les problèmes 
d'ordre des frames dans les vidéos, avec diverses méthodes basées sur
l'analyse des caractéristiques visuelles et le suivi d'objets.
"""

__version__ = '0.1.0'
__author__ = 'CLD'

from .video_processor import VideoProcessor

__all__ = ['VideoProcessor']