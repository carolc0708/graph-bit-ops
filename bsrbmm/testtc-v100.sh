#!/bin/bash

make tc-v100

for filename in ../data/*.mtx
do
  echo "========================================="
  ./tc4v $filename
  echo "-----------------------------------------"
  ./tc8v $filename
  echo "-----------------------------------------"
  ./tc16v $filename
  echo "-----------------------------------------"
  ./tc32v $filename
done