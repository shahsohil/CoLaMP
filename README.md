# CoLaMP

## Introduciton

This is a MATLAB implementation based of the CoLaMP algorithms presented in the following paper ([download](http://www.cv-foundation.org/openaccess/content_cvpr_2016/papers/Shah_Estimating_Sparse_Signals_CVPR_2016_paper.pdf)).
Estimating Sparse Signals with Smooth Support via Convex Programming and Block Sparsity
Sohil Atul Shah, Christoph Studer and Tom Goldstein

The source code and datasets are published under the MIT Licence. See [LICENSE](LICENSE) for details. In general you can use them for any purpose with proper attibution. If you do something interesting with the code, we'll be happy to know about it. Feel free to contact us.

## Running Algorithms
As presented in the paper Algorithm 1 is implemented as part of [block-sparse-RPCA](robustpca_l2sparsity_mxm_block_splitting.m) code and Algorithm 2 is implemented in [matching_pursuit.m](matching_pursuit.m). Apart from this, we have also provided ADMM implementation for Algorithm 1 [here](robustpca_l2sparsity_mxm_block.m). The denoising algorithm is implemented by plugging in Primal-Dual algorithm of this [paper](http://www.cv-foundation.org/openaccess/content_cvpr_2014/papers/Ono_Decorrelated_Vectorial_Total_2014_CVPR_paper.pdf).   

We have included test wrapper for three of the four application in the codebase for the ease of understanding the input and output and for reproducing some of the paper's qualitative results. We will soon upload wrapper for denoising application. 
