# TODO – Learn2Slither

- [x] Basculer `main.py` sur la logique valide (fait, à vérifier visuellement)
- [x] Ajouter configuration flake8 dans `pyproject.toml` (fait)
- [x] Compatibilité SDL macOS/Linux (fait: auto via `main.py`)
- [x] Créer package `learn2slither/` (env, state, actions, rewards, q_agent, trainer, sdl, config)
- [x] Déplacer `SnakeEnv` (logique pure) dans `learn2slither/env.py`
- [x] Déplacer `get_state`/debug dans `learn2slither/state.py` (encodeur minimal)
- [x] Isoler rendu Pygame dans `learn2slither/render.py`
- [x] Isoler contrôles dans `learn2slither/controls.py`
- [x] Implémenter `q_agent` (Q-table, ε-greedy, save/load, gel apprentissage)
- [x] Implémenter `rewards` et encoder d’état (danger_* + directions)
- [x] Implémenter `trainer` (boucle d’épisodes, headless support)
- [x] Ajouter CLI (`--train`, `--verbose`, `--save-model`, `--load-model`, `--save-every`, `--vision`)
- [x] Ajouter tests unitaires (env init/collisions/spawn, agent update, state encoder)
- [ ] Nettoyer `test.py` ou le retirer (Linux-only)
- [ ] Mettre à jour README (uv run, options CLI) 