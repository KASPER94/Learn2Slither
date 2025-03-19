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
