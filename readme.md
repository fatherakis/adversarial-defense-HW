# MSc Thesis - Adversarial Robustness Strategies for Neural Networks in Hardware

This repository contains my thesis Jupyter Notebooks and Scripts along with the appropriate links for most generated files due to their size. It does not serve as a complete guide to recreate this work however most scripts and notebooks should be  complete to run on any computationally capable system.

My thesis is available here:

``http://dx.doi.org/10.26240/heal.ntua.28198``
OR
``http://artemis.cslab.ece.ntua.gr:8080/jspui/handle/123456789/19042``


All compiled and generated files are available in this drive link:\
https://t.ly/_ASab  **Yes it's a drive link, I used a shortener (t.ly) so it won't be too long.**


> [!Important]
> These files are **mandatory** to use the scripts. The folders are oranized per model per dataset with clear labeling for all model states.


A summary on this repo and the scope of my thesis:

```
Convolutional Neural Networks while highly effective for pattern recognition, are vulnerable to small perturbations
in the input (image) leading to erroneous predictions. This issue is further amplified in hardware implementations
due to precision limitations and requirements like quantization. In this thesis we propose a series of adversarial
defense strategies in order to enhance the robustness of neural networks specifically in hardware applications.
All in all, through this process a 45% increase in robustness is achieved.
```

Mostly python and jupyter notebooks are used with the PyTorch Framework. For the hardware, we use Xilinx/AMD Versal SoC VCK190 board with its accompanied VitisAI 3.0 software for compilation and support. Lastly, credits for all libraries, pretrained models and resourses are also inlcuded in the thesis references section.

The proposed strategies are applied to the following models and datasets:

* ResNet20

* ResNet56

* MobileNetV2

* ***CIFAR-10***

* ***FashionMNIST***


Essentialy we have 6 different model variations to evaluate: Each of the 3 models with each dataset.


We will be running the following proceedures:

* Adversarial Example Generation through a black-box algorithm to simulate a real attack as close as possible (Hop-Skip-Jump Attack)

* Model Training on adversarial examples, known as Adversarial Training, progresively increasing robustness and generalization

* Quantization of model, to benefit from quantization effects and estimate as close as possible the real results from the hardware device, regarding the model's robustness.

* Repetition of the above steps 2 more times with smaller and smaller adversarial dataset sizes to further take advantage of Adversarial Training effects.

* Quantize and evaluate models through VitisAI

* Finally, deploy on hardware device (Versal VCK190) and examine results.


> [!Note]
> There are many similarities with another repository of mine: [AOHW-225](https://github.com/fatherakis/AOHW-225) which was for the AMD Open Hardware Competition. That repository contains fully step-by-step instructions for **1 of the 6** model-dataset combinations. Specifically its for MobileNetV2 and CIFAR-10. However, the process is the exact same for everything except some small adjustments or in the case of FashionMNIST versions, major model and dataset changes (the training code is still the same).
If you are interested in running anything, please check that [guide there](https://github.com/fatherakis/AOHW-225/readme.md) and then adjust accordingly with the resources here.

## Adversarial Example Generation

There are 4 Notebooks:

* [MobileNet FashionMNIST Generation](resources\adversarial_example_generation\mobilenet_fashionmnist_gen.ipynb)

* [ResNet20 FashionMNIST Generation](resources\adversarial_example_generation\resnet20_fashionmnist_gen.ipynb)

* [ResNet56 FashionMnist Generation](resources\adversarial_example_generation\resnet56_fashionmnist_gen.ipynb)

* [CIFAR Generation](resources\adversarial_example_generation\cifar_gen.ipynb)


Note that there are seperate files for FashionMNIST and one singular for CIFAR-10 for adversarial examples. In the case of CIFAR-10 there are no changes required except of the model name or model file loading which can be easily changed based on preferences (Models are pre-trained on CIFAR-10). In the case of FashionMNIST, each model class is defined manually in order to change functions and layers appropriately to support the dataset.
Thus, there are seperate files on which you can generate said examples.

> [!Caution]
> FashionMNIST Notebooks require the dataset files available in the Google Drive link. Furthermore, the notebooks as-is will be functional only for the first adversarial example set. After this, the model should be retrained and exported. To generate the new set of adversarial examples, the retrained model should be loaded instead of the initial one. This applies to both CIFAR-10 and FashionMNIST versions.


## Adversarial Training

Re-training procedure is fairly straight forward. For CIFAR-10 models we use [this notebook](/resources/adversarial_training/model-retrain.ipynb) and for FashionMNIST [this one](/resources/adversarial_training/fashion_model_retrain.ipynb). Note that the correct model state should be loaded for each training iteration. Also training functions have some accuracy threshold flags which can be changed. These were put for consistency through testing. Finally, for FashionMNIST models, the provided notebook only accounts for the MobileNetV2 architecture. In order to retrain let's say ResNet20 on FashionMNIST examples, one could simply copy the class from the adversarial example generation notebook and continue as usual.

## Quantization

After a re-training process, the model should be quantized and generate a small sample of adversarial examples to evaluate any changes in robustness and behavior. Either way, in order to run a model on our Versal device, it needs to be quantized, so evaluating our results this way will also give an image on what to expect. [Quantization CIFAR-10](/resources/quantization/quantization-hopskip.ipynb) does just that, for CIFAR-10 models. Note that for quantization, all models need small changes in their functions to work. The particular notebook is setup for ResNet56 after the first re-training iteration to also function as an example on how to load the model state and files after retraining. Quantization doesn't change much between models and datasets, it runs the same quantizer and generates a small adversarial example set for later evaluation. [Quantiezed Definitions](/resources/quantization/quantized_model_def.ipynb) notebook contains all model definitions for both datasets along with the quantization procedure. Simply copy paste the desired model-dataset combo and generate adversarial examples.



> [!Note]
> If anyone intends to run these notebooks, be very careful with the file paths.

## Hardware Compiling and Platform


At this point, the work as far as models go is complete, all models can be easily evaluated, attacked, retrained and quantized. Notebooks for statistics and evaluations are simply ommited since they serve no purpose in this repository. What remains however is VitisAI and Versal procedures.

These are the exact same as the [AOHW-225](https://github.com/fatherakis/AOHW-225) project (project only focuses on MobileNet CIFAR-10) so I highly suggest checking the fully detailed guide there. I won't repeat the same steps or provide duplicate files.

In order however to properly compile the finalized models in VitisAI, the models must be defined in their quantization-compatible form and initialized with the appropriate state. For this reason, while command arguments and steps are the same as the other sub-project, all required python scripts to compile the models for Versal are provided in [Vitis AI Models dir](/resources/vitis_ai_models/). Be sure to verify paths and file names.


Finally, after VitisAI compilation, models are exported in xmodel format readable by Versal. At that point they are transfered onto Versal along with the appropriate scripts available at [Versal Directory](/resources/versal_scripts/) to run and produce the finalized results. More information about the compilation and execution proccedure can be found on the very detailed guide here:  [Vitis AI Image Classification Guide](https://github.com/fatherakis/Vitis-AI-Image-CNN-Guide). In this stage, the differences between CIFAR-10 and FashionMNIST are all accounted for, requiring only the appropriate model files in the corresponding folders, letting the scripts handle the rest. 
