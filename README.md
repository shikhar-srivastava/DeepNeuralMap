# DNM
Implementation of Deep Neural Maps paper (ICLR 2018) [TF]

## Main File: neural_map.py

All map computation, gradient calculation is done in TF.math. Optimized for @tf.function use in TF 2.x.
Can be exported to TF 1.x as well with minor modifications.

Map Viz is done through tensorboard.
