#!/bin/bash

source ./cifar10/run_all_cifar10_target.sh main vck190 ./resnet20CIFAR.xmodel
source ./cifar10/run_all_cifar10_target.sh main vck190 ./resnet56CIFAR.xmodel
source ./cifar10/run_all_cifar10_target.sh main vck190 ./mobilenetCIFAR.xmodel

source ./fashion/run_all_fashionMNIST_target.sh main vck190 ./resnet20Fashion.xmodel
source ./fashion/run_all_fashionMNIST_target.sh main vck190 ./resnet56Fashion.xmodel
source ./fashion/run_all_fashionMNIST_target.sh main vck190 ./mobilenetFashion.xmodel
