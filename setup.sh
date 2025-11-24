# !/bin/bash
# FastWER calculation
pip install -r requirements.txt
git clone https://github.com/PRHLT/fastwer.git
pip install ./fastwer

# BEER installation
wget https://raw.githubusercontent.com/stanojevic/beer/master/packaged/beer_2.0.tar.gz
tar xfvz beer_2.0.tar.gz
rm beer_2.0.tar.gz