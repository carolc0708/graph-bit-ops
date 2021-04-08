#!/bin/bash

make tc-p100

for filename in ../data/*.mtx
do
  echo "========================================="
  ./tc4p $filename
  echo "-----------------------------------------"
  ./tc8p $filename
  echo "-----------------------------------------"
  ./tc16p $filename
  echo "-----------------------------------------"
  ./tc32p $filename
done