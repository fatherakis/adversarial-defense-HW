#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


# T-shirt Trouser Pullover Dress Coat Sandal Shirt Sneaker Bag Ankle-boot

tar -xvf test.tar &> /dev/null
#mv ./build/dataset/cifar10/test ./test
#rm -r ./build
cd ./test

cd T-shirt
mv *.png ../
cd ..
rm -r T-shirt/

cd Trouser
mv *.png ../
cd ..
rm -r Trouser/

cd Pullover
mv *.png ../
cd ..
rm -r Pullover/

cd Dress
mv *.png ../
cd ..
rm -r Dress/

cd Coat
mv *.png ../
cd ..
rm -r Coat/

cd Sandal
mv *.png ../
cd ..
rm -r Sandal/

cd Shirt
mv *.png ../
cd ..
rm -r Shirt/

cd Sneaker
mv *.png ../
cd ..
rm -r Sneaker/

cd Bag
mv *.png ../
cd ..
rm -r Bag

cd Ankle-boot
mv *.png ../
cd ..
rm -r Ankle-boot

cd ..
