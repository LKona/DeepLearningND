# Dog-Breed-Classifier
Udacity's Deep Learning Nanodegree Project.

This is a Convolutional Neural Networks (CNN) project in Pytorch. 

Objective: Given an image of a dog, identify an estimate of the canine’s breed. If supplied an image of a human, the code will identify the resembling dog breed.

This was a fun project from which I learned a lot. Some of the things I learned are:
  1. How to build a pipeline to process real-world, user-supplied images. 
  2. Exploring state-of-the-art CNN models for classification.
  3. Making important design decisions about the user experience for the app.
  4. Understand the challenges involved in piecing together a series of models designed to perform various tasks in a data    processing pipeline. Each model has its strengths and weaknesses, and engineering a real-world application often involves solving many problems without a perfect answer.
  5. How to tweak hyperparameters for optimal results.

Summary of Implementation:
  1. The model I created from scratch attained 13% accuracy.
    Net(
    (conv1): Conv2d(3, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (conv2): Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
    (conv3): Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (conv4): Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
    (pool): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
    (fc1): Linear(in_features=2304, out_features=512, bias=True)
    (fc2): Linear(in_features=512, out_features=256, bias=True)
    (fc3): Linear(in_features=256, out_features=133, bias=True)
    (dropout): Dropout(p=0.3)
  )
  2. For Transfer Learning, I used VGG-16 and replaced its classifier with my own that has 4 fully connected layers, each followed by a ReLU and Dropout of 0.3. This model's accuracy was 63% after about 8 epochs of training.
  
  I figured I could use a model like Resnet50 for better accuracy.(Which is what I am working on next as an improvement for this project)
