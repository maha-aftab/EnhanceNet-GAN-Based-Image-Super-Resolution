# Super-Resolution-GAN
## Introduction 
The web app aims to generate a Super Resolution image of your low resolution image using Generative Adversarial Network. Various techniques are adopted for image upscaling which results in distorted image or reduced visual quality images. Deep Learning provides better solution to get optimized images. Super Resolution Genarative Adversarial Network is one amoung them. 

<!-- ![alt text](demo.gif) -->
## GAN
Generative Adversarial Networks, or GANs for short, are an approach to generative modeling using deep learning methods, such as convolutional neural networks. Generative modeling is an unsupervised learning task in machine learning that involves automatically discovering and learning the regularities or patterns in input data in such a way that the model can be used to generate or output new examples that plausibly could have been drawn from the original dataset.GANs will have two different networks Generator and a Discriminator. Generator generates the data, which is cross checked by the discriminator. The loss found is rectifided through backpropagation. SR-GAN downsamples the high resolution images to create Low resolution images for training and Generater generates super resolution images and that is cross checked by the Discriminator.
 
 
