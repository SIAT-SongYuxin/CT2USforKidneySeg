# CT2USforKidneySeg
CT2US：Cross-modal Transfer Learning for Kidney Segmentation in Ultrasound Images with Limited Data

# Paper
Submission to MICCAI2021:"CT2US：Cross-modal Transfer Learning for Kidney Segmentation in Ultrasound Images with Limited Data"

Abstract.Accurate segmentation of kidney in ultrasound images is a vital procedure in clinical diagnosis and interventional operation. In recent years, deep learning technology has demonstrated promising prospects in medical image analysis. However, due to the inherent problems of ultrasound images, the data with annotations is scarce and arduous to acquire, hampering the application of the data-hungry deep learning methods. In this paper, to solve the lack of training data, we propose the cross-modal transfer learning from CT to US with leveraging the annotated data in the CT modality. Particularly, we adopt CycleGAN to synthesize US images from CT data and construct the transition dataset to mitigate the immense domain discrepancy between US and CT. Mainstream CNN networks such as FCN, U-Net are pre-trained on the transition dataset and then transferred to the real US images. Experimental results reveal that our approach effectively improves the accuracy and generalization ability in cross-site test with limited training data.

# Installation
