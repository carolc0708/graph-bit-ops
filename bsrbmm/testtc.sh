#!/bin/bash

make tc

for filename in ../../data/*.mtx
do
  echo "========================================="
  ./tc4 $filename
  echo "-----------------------------------------"
  ./tc8 $filename
  echo "-----------------------------------------"
  ./tc16 $filename
  echo "-----------------------------------------"
  ./tc32 $filename
done