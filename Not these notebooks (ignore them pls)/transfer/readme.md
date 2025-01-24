# Transfer learning
I got these notebooks from a [TensorFlow tutorial](https://www.tensorflow.org/tutorials/images/transfer_learning_with_hub) and made a copy. 

Transfer learning is a technique in machine learning in which knowledge learned from a task is re-used in order to boost performance on a related task. For example, for image classification, knowledge gained while learning to recognize cars could be applied when trying to recognize trucks. ([1](https://en.wikipedia.org/wiki/Transfer_learning))

For this transfer learning part, we used the MobileNetV2 model, which is a pre-trained model on the [ImageNet](https://image-net.org/about.php) dataset. We used the model to classify images of 10 types of animals from the [Animals-10 dataset](https://www.kaggle.com/datasets/alessiocorrado99/animals10/data). 

## How to run
The [transfer learning notebook](./transfer_learning%20real.ipynb) Has three parts: 
- Getting the data, preprocessing it
- Loading the trained model (MobileNetV2 + classification layer + trained on Animals-10 dataset) and testing it on the test set
- Training the model (MobileNetV2 + classification layer) on the Animals-10 dataset. 

The first part needs to be run first by anyone who wants to run the notebook. The second part can be run by anyone who wants to test the model. The third part can be run by anyone who wants to train the model again. If you are going to train the model again, please turn on the GPU acceleration option, but I'd recommend simply using the second option to test the model. 

The notebook was originally written to be run on Google Colab, so the model loading part is written to load the model from Google Drive. If you are going to run the notebook on your local machine, you can simply change the path to the model file. The [model file](./transfer_model2.keras) is available in this directory.

You can click the button at the beginning to run the notebook on colab. Also, If you are going to train the model again, please turn on the GPU acceleration option. 
