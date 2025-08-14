# Learn2Slither

42 set up:
    # 1. Installer Miniconda
    MYPATH="/goinfre/$USER/miniconda3"
    curl -LO "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    sh Miniconda3-latest-Linux-x86_64.sh -b -p $MYPATH
    $MYPATH/bin/conda init bash
    $MYPATH/bin/conda config --set auto_activate_base false
    source ~/.bashrc

    # 2. Créer l'environnement Conda
    conda create --name 42AI-$USER python=3.7 jupyter pandas pycodestyle numpy -y
    conda activate 42AI-$USER
    pip install pygame

    # 3. Copier la bonne version de libstdc++.so.6 dans Conda
    cp /usr/lib/x86_64-linux-gnu/libstdc++.so.6* /goinfre/$USER/miniconda3/envs/42AI-$USER/lib/

    # 4. Ajouter les variables d'environnement au .bashrc
    echo 'export LD_LIBRARY_PATH=/goinfre/$USER/miniconda3/envs/42AI-$USER/lib:$LD_LIBRARY_PATH' >> ~/.bashrc
    echo 'export SDL_VIDEODRIVER=x11' >> ~/.bashrc
    echo 'conda activate 42AI-$USER' >> ~/.bashrc
    source ~/.bashrc

    # 5. Vérifier que tout fonctionne
    python3 -c "import pygame; print('Pygame OK')"
    echo $LD_LIBRARY_PATH
    echo $SDL_VIDEODRIVER
    conda info --envs

    # 6. Tester Learn2Slither
    python3 test.py


Explication Algo QN:
ε-greedy (exploration vs exploitation)
    - But: parfois explorer (essayer une action aléatoire) pour découvrir mieux,    parfois exploiter (choisir la meilleure action connue).
    - Règle: avec probabilité ε → action aléatoire; sinon → action argmax des Q-valeurs.
    Ce que n’est pas ε: ce n’est pas une remise/discount. Le “discount” du futur est géré par γ, pas par ε.
    - Durée courte/longue: la décision “greedy” maximise Q(s, a) courant, mais Q(s, a) lui-même approxime une somme de récompenses futures actualisées par γ; donc l’action “greedy” tient déjà compte du long terme via γ.
    - Décroissance: ε démarre haut (explore), puis décroît vers ε_min pour exploiter de plus en plus.

Q-learning: Q[s,a] ← Q[s,a] + α × (r + γ × max_a' Q[s',a'] − Q[s,a])
    - s, a, r, s': état courant, action prise, récompense reçue, nouvel état.
    - α (learning rate): taille du pas de mise à jour. Petit α = apprentissage lent mais stable.
    - γ (discount): pondère l’importance du futur: 0 = myope (seulement r immédiat); proche de 1 = long terme important.
    - max_a' Q[s', a']: meilleure valeur estimée au prochain état (off-policy; on suppose qu’on agira de façon optimale ensuite).
    - TD error δ: δ = r + γ max_a' Q[s',a'] − Q[s,a]. On ajuste Q[s,a] de α×δ vers la cible.
    - Intuition: on rapproche progressivement Q de la cible “récompense immédiate + valeur future”.

