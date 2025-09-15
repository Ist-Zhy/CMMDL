# CMMDL
Codes for **"Cross-Modal Multi-Domain Learning for Image Fusion"**

## Abstract
The rapid development of deep learning provides an excellent solution for end-to-end multi-modal image fusion. However, existing methods mainly focus on the spatial domain and fail to fully utilize valuable information in the frequency domain. Moreover, even if spatial domain learning methods can optimize convergence to an ideal solution, there are still significant differences in high-frequency details between the fused image and the source images. Therefore, we propose a Cross-Modal Multi-Domain Learning (CMMDL) method for image fusion. Firstly, CMMDL employs the Restormer structure equipped with the proposed Spatial-Frequency domain Cascaded Attention (SFCA) mechanism to provide comprehensive and detailed pixel-level features for subsequent multi-domain learning. Then, we propose a dual-domain parallel learning strategy. The proposed Spatial Domain Learning Block (SDLB) focuses on extracting modality-specific features in the spatial domain through a dual-branch invertible neural network, while the proposed Frequency Domain Learning Block (FDLB) captures continuous and precise global contextual information using cross-modal deep perceptual Fourier transforms. Finally, the proposed Heterogeneous Domain Feature Aggregation Block (HDFAB) promotes feature interaction and fusion between different domains through various pixel-level attention structures to obtain the final output image. Extensive experiments demonstrate that the proposed CMMDL achieves state-of-the-art performance on multiple datasets.

## Dataset
Download the data and place the data according to the path in './data/'.
"MSRS"

## Training
At the top of the CMMDL_train.py file, set your weights and dataset paths, then run the following code:
torchrun --nnodes=1 --nproc_per_node 2 CMMDL_train.py

## Testing
Run the following command for testing:
python CMMDL_test.py
