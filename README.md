# DNM
Implementation of Deep Neural Maps paper (ICLR 2018) [TF]

## Main File: neural_map.py

All map computation, gradient calculation is done in TF.math. Optimized for @tf.function use in TF 2.x.
Can be exported to TF 1.x as well with minor modifications. Architecture from paper has been extended to N-Dims here. 

Please refer to DNM paper here: https://arxiv.org/pdf/1810.07291.pdf


