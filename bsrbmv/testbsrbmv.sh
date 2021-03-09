#!/bin/bash

make

for filename in ../*.mtx
do
  echo "========================================="
  ./bsrbmv8 $filename
  echo "-----------------------------------------"
  ./bsrbmv16 $filename
  echo "-----------------------------------------"
  ./bsrbmv32 $filename
  echo "-----------------------------------------"
  ./bsrbmv64 $filename
done