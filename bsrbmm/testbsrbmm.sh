#!/bin/bash

make

for filename in ../*.mtx
do
  echo "========================================="
  ./bsrbmm8 $filename
  echo "-----------------------------------------"
  ./bsrbmm16 $filename
  echo "-----------------------------------------"
  ./bsrbmm32 $filename
  echo "-----------------------------------------"
  ./bsrbmm64 $filename
done