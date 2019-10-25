# frequency-aware-conv2d-layer-pytorch

A Pytorch implementation of frequency-aware convolutional 2D layer.

pytorch 1.1

## Usage

Instead of `nn.Conv2d()`, use `FreqAwareConv2dLinearBiasOffset()`. 
The API of `FreqAwareConv2dLinearBiasOffset()` is same as that of `nn.Conv2d` as of pytorch 1.1.

## Reference

[Acoustic Scene Classification and Audio Tagging with Receptive-Field-Regularized CNNs](https://www.researchgate.net/publication/334250606_CP-JKU_Submissions_to_DCASE'19_Acoustic_Scene_Classification_and_Audio_Tagging_with_Receptive-Field-Regularized_CNNs) by Khaled Koutini, Hamid Eghbal-Zadeh, and Gerhard Widmer, 2019, DCASE workshop. 