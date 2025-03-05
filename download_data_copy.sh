#!/bin/bash


mkdir ./data/downloaded/SPair-71k

echo -e "\e[1mGetting SPair-71k data\e[0m"
cd ./data/downloaded/SPair-71k
wget http://cvlab.postech.ac.kr/research/SPair-71k/data/SPair-71k.tar.gz
tar xzf SPair-71k.tar.gz
mv SPair-71k/* ./
echo -e "\e[32m... done\e[0m"