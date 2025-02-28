# Learn2Slither

42 set up:
    $> MYPATH="/goinfre/$USER/miniconda3"
    $> curl -LO "https://repo.anaconda.com/miniconda/ Miniconda3-latest-Linux-x86_64.sh"
    $>sh Miniconda3-latest-Linux-x86_64.sh -b -p $MYPATH
    $> $MYPATH/bin/conda init bash
    $> $MYPATH/bin/conda config --set auto_activate_base false
    $> source ~/.bash_profile
    $> conda create --name 42AI-$USER python=3.7 jupyter pandas pycodestyle numpy pygame
    $> conda activate 42AI-skapersk