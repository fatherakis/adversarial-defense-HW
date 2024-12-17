#!/bin/bash

# Copyright © 2023 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT

# Author: Daniele Bagni, Xilinx Inc
# date:  28 Apr. 2023


TARGET=$2
#vek280
MODELFILE=$3

#clean
clean_fashion(){
echo " "
echo "clean fashionMNIST"
echo " "
cd fashion
rm -rf test
rm -f *~
rm -f  run_cnn cnn* get_dpu_fps *.txt
rm -rf rpt
rm -f  *.txt
rm -f  *.log
mkdir -p rpt
cd ..
}

# compile CNN application
compile_fashion(){
echo " "
echo "compile fashion"
echo " "
cd fashion/code
echo "PWD1 = " $PWD
bash -x ./build_app.sh
mv code ../cnn_fashion # change name of the application
bash -x ./build_get_dpu_fps.sh
mv code ../get_dpu_fps
cd ../..
echo "PWD2 = " $PWD
}

# build cifar10 test images
test_images_fashion(){
echo " "
echo "build test images for fashion"
echo " "
cd fashion
bash ./build_fashionMNIST_test.sh
cd ..
echo " "
echo "PWD3 = " $PWD
}

# now run the cifar10 classification with 4 CNNs using VART C++ APIs
run_cnn_fahion(){
echo " "
echo " run fashion CNN"
echo " "
cd fashion
./cnn_fashion ./${MODELFILE}.xmodel ./test/ ./fashionMNIST_labels.dat | tee ./rpt/predictions_fashion_${MODELFILE}.log
# check DPU prediction accuracy
bash -x ./fashionMNIST_performance.sh ${TARGET} ${MODELFILE}
echo "PWD4 = " $PWD
cd ..
}

#remove images
end_fashion(){
echo " "
echo "end of fashion"
echo " "
cd fashion
rm -rf test
cd ../
echo "PWD5 = " $PWD
#tar -cvf target.tar ./target_*
}


main()
{
    #clean_fashion
    #compile_fashion
    test_images_fashion
    run_cnn_fahion
    end_fashion
}




"$@"
