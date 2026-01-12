# !/bin/bash

conda env create -f environment.yaml
eval "$(conda shell.bash hook)"
conda activate deep

# fastwer installation
git clone https://github.com/PRHLT/fastwer.git
pip install ./fastwer

# BEER installation
wget https://raw.githubusercontent.com/stanojevic/beer/master/packaged/beer_2.0.tar.gz
tar xfvz beer_2.0.tar.gz
rm beer_2.0.tar.gz

echo "################################"
echo "Setup complete."
echo "Activate the conda environment to use the app:"
echo "  conda activate deep"