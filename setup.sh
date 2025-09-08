# !/bin/bash
# FastWER calculation
pip install -r requirements.txt
git clone https://github.com/PRHLT/fastwer.git
cd fastwer
python setup.py install
cd ..

# BEER installation
wget https://raw.githubusercontent.com/stanojevic/beer/master/packaged/beer_2.0.tar.gz
tar xfvz beer_2.0.tar.gz
rm beer_2.0.tar.gz