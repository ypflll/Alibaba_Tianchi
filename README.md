# Alibaba_Tianchi
This is part of code for Alibaba Tianchi AI competition: https://tianchi.aliyun.com/competition/introduction.htm?raceId=231601.


The code is mainly based on Julian de Wit's code on kaggle dsb2017 (second place): https://github.com/juliandewit/kaggle_ndsb2017.


We also refer to the dsb2017 tutorial: https://www.kaggle.com/c/data-science-bowl-2017#tutorial.


Our team (DeepLuna) ranks 71/2887 in the competition. We trained a CNN to detect the nodules in lung CT images. The input of the CNN is cubes from 3d CT image, and it will predict the probability that the cube includes a nodule. But the result is not very satified mainly because the coordinate of the nodule is not very accurate. If have time, we recommand you use a 3D-unet and a false positive reduction net to do this job, like the 9th place in dsb2017: https://eliasvansteenkiste.github.io/machine%20learning/lung-cancer-pred/


This repository includes all pre-process code like: coordinate exchange, lung mask segmentation, plot 3d figure, get positive samples and negative samples for the CNN. It also includes the predict code. 


Code to train the CNN is here: https://github.com/CyranoChen/deepluna.

If you have a validate dataset, you can see the result by running the evaluation code: https://luna16.grand-challenge.org/evaluation/.



