"""Utilitaires de configuration SDL pour Pygame.

Permet d'initialiser les variables d'environnement SDL selon l'OS ou en mode
headless (CI). Ne réalise aucun import Pygame pour rester indépendant.
"""

from __future__ import annotations

import os
import sys
from typing import Optional


def configure_sdl(headless: bool = False, *, force_driver: Optional[str] = None) -> None:
    """Configure les variables d'environnement SDL.

    - macOS: "cocoa"
    - Linux: "x11"
    - headless: "dummy" (désactive l'affichage)

    Paramètres
    ----------
    headless: bool
        Si True, force le driver vidéo à "dummy".
    force_driver: Optional[str]
        Si fourni, utilise ce driver (prioritaire sur la détection auto).
    """
    if force_driver:
        os.environ["SDL_VIDEODRIVER"] = force_driver
    elif headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"
    else:
        if "SDL_VIDEODRIVER" not in os.environ:
            if sys.platform.startswith("darwin"):
                os.environ["SDL_VIDEODRIVER"] = "cocoa"
            elif sys.platform.startswith("linux"):
                os.environ["SDL_VIDEODRIVER"] = "x11"

    # Rendu logiciel, utile sur CI/VM
    os.environ.setdefault("SDL_RENDER_DRIVER", "software")
    # Variables Mesa (inoffensives hors Linux)
    os.environ.setdefault("LIBGL_ALWAYS_SOFTWARE", "1")
    os.environ.setdefault("MESA_LOADER_DRIVER_OVERRIDE", "swrast") 