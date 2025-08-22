import os
import sys

# Sélection robuste du backend Matplotlib avant pyplot
# - macOS: tenter 'MacOSX', puis 'TkAgg'
# - autres: tenter 'TkAgg', puis 'QtAgg'
# - fallback: 'Agg' (headless)
BACKEND = os.environ.get('MPLBACKEND')

try:
    import matplotlib as mpl  # import léger, sans pyplot
    backend_set = False

    if not BACKEND:
        preferred = ['MacOSX', 'TkAgg'] if sys.platform.startswith('darwin') else ['TkAgg', 'QtAgg']
        for candidate in preferred:
            try:
                if candidate == 'TkAgg':
                    import tkinter  # Vérifie la dispo de Tk
                mpl.use(candidate, force=True)
                BACKEND = candidate
                backend_set = True
                break
            except Exception:
                continue
        if not backend_set:
            mpl.use('Agg', force=True)
            BACKEND = 'Agg'
    else:
        # Respecte le backend imposé par l'environnement
        mpl.use(BACKEND, force=True)
except Exception:
    # En dernier recours, utilise Agg
    import matplotlib as mpl  # noqa: F401
    BACKEND = 'Agg'

import matplotlib.pyplot as plt
from IPython import display

plt.ion()


def plot(scores, mean_scores):
    """Trace la courbe d'entraînement.

    - En mode interactif (GUI dispo): affiche la figure
    - En mode headless ('Agg'): sauvegarde dans training_plot.png
    """
    # Efface et met à jour la figure
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    if scores:
        plt.text(len(scores)-1, scores[-1], str(scores[-1]))
    if mean_scores:
        plt.text(len(mean_scores)-1, mean_scores[-1], str(mean_scores[-1]))

    if (BACKEND or '').lower() == 'agg':
        # Mode headless: on enregistre l'image au lieu d'ouvrir une fenêtre
        out = os.environ.get('L2S_PLOT_PATH', 'training_plot.png')
        plt.savefig(out)
    else:
        # Mode interactif
        plt.show(block=False)
        plt.pause(.1)