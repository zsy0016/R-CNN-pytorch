#/bin/bash
wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/17flowers.tgz
mkdir data/17flowers
tar xvf 17flowers.tgz -C data/17flowers
rm 17flowers.tgz

wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/trimaps.tgz
tar xvf trimaps.tgz -C data
rm trimaps.tgz

wget http://www.robots.ox.ac.uk/~vgg/data/flowers/17/datasplits.mat
mv datasplits.mat data

