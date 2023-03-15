## MoCo Demo: CIFAR-10

This is a simple demo for training MoCo on CIFAR-10. It can be run directly in a Colab notebook using a publicly available GPU.

#### Results

These are the ResNet-18 classification accuracy of a **kNN monitor** on the unsupervised pre-training features. 

| config | 200ep | 400ep | 800ep |
| --- | --- | --- | --- |
| Asymmetric | 82.6 | 86.3 | 88.7 |
| Symmetric | 85.3 | 88.5 | 89.7 |

#### Notes

* **Symmetric loss**: the original MoCo paper uses an *asymmetric* loss -- one crop is the query and the other crop is the key, and it backpropagates to one crop (query). Following SimCLR/BYOL, here we provide an option of a *symmetric* loss -- it swaps the two crops and computes an extra loss. The symmetric loss behaves like 2x epochs of the asymmetric counterpart: this may dominate the comparison results when the models are trained with a fixed epoch number.

* **SplitBatchNorm**: the original MoCo was trained in 8 GPUs. To simulate the multi-GPU behavior of BatchNorm in this 1-GPU demo, we provide a SplitBatchNorm layer. We set `--bn-splits 8` by default to simulate 8 GPUs. `--bn-splits 1` is analogous to SyncBatchNorm in the multi-GPU case.

* **kNN monitor**: this demo provides a kNN monitor on the test set. Training a linear classifier on frozen features often achieves higher accuracy. To train a linear classifier (not provided in this demo), we recommend using lr=30, wd=0, epochs=100 with a stepwise or cosine schedule. The ResNet-18 model (kNN 89.7) has 90.7 linear classification accuracy.

#### Disclaimer

This demo aims to provide an interface with a free GPU (thanks to Colab) for understanding how the code runs. We suggest users be careful to draw conclusions from CIFAR, which may not generalize beyond this dataset. We have verified that it is beneficial to have the momentum encoder (disabling it by `--moco-m 0.0` should fail), queue size (saturated at `--moco-k 4096`) and ShuffleBN (without which loses 4% at 800 epochs) on CIFAR, similar to the observations on ImageNet. But new observations made only on CIFAR should be judged with caution.

#### References
This demo is adapted from:
* http://github.com/zhirongw/lemniscate.pytorch
* https://github.com/leftthomas/SimCLR

