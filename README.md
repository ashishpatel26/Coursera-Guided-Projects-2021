# **Coursera Guided Projects**

## [Computer Vision - Object Detection with OpenCV and Python](https://github.com/shejz/Coursera-guided-projects/tree/master/Computer%20Vision%20-%20Object%20Detection%20with%20OpenCV%20and%20Python)

### **Project Structure**
- Face Detection
- Eyes Detection
- Face and Eyes Detection
- Pedestrians Detection
- Cars Moving Detection
- Car's Plate Detection

## [Movie Recommendation System using Collaborative Filtering](https://github.com/shejz/Coursera-guided-projects/tree/master/Movie%20Recommendation%20System%20using%20Collaborative%20Filtering)
In this project-based course, we will create a recommendation system using Collaborative Filtering with help of `Scikit-surprise` library, which learns from past user behavior. We will be working with a movie lense dataset and by the end of this project, we will be able to give unique movie recommendations for every user based on their past ratings.

## [Medical Diagnosis using Support Vector Machines](https://github.com/shejz/Coursera-guided-projects/tree/master/Medical%20Diagnosis%20using%20Support%20Vector%20Machines)
The dataset we are going to use comes from the **National Institute of Diabetes and Digestive and Kidney Diseases**, and contains anonymized diagnostic measurements for a set of female patients.  We will train a **support vector machine** to predict whether a new patient has diabetes based on such measurements.

## [Avoid Overfitting Using Regularization in TensorFlow](https://github.com/shejz/Coursera-guided-projects/tree/master/Avoid%20Overfitting%20Using%20Regularization%20in%20TensorFlow)
In this project, you will learn the basics of using weight regularization and dropout regularization to reduce over-fitting in an image classification problem. By the end of this project, you will have created, trained, and evaluated a Neural Network model that, after the training and regularization, will predict image classes of input examples with similar accuracy for both training and validation sets.

When we train neural network models, you may notice the model performing significantly better on training data as compared to data that it has not seen before, or not trained on before. This means that while we expect the model to learn the underlying patterns from a given data-set, often the model will also memorize the training examples. It will learn to recognize patterns which may be anomalous or may learn the peculiarities in the data-set. This phenomenon is called over-fitting and it's a problem because a model which is over-fit to the training data will not be able to generalize well to the data that it has not seen before and that sort of defeats the whole point of making the model learn anything at all. We want models which are able to give us predictions as accurately on new data as they can for the training data.

- Develop an understanding on how to avoid over-fitting with weight regularization and dropout regularization
- Be able to apply both weight regularization and dropout regularization in Keras with TensorFlow backend

**Regularization**

One of the reasons for **over-fitting** is that some of these parameter values can become somewhat large and therefore become too influential on the linear outputs of various hidden units and subsequently become too influential on the non-linear outputs from the activation functions as well. And it can be observed that by regularizing the weights in a way that their values don't become too large, we can reduce the over-fitting. In dropouts, by randomly removing certain nodes in a model, we are forcing the model to NOT assign large values to any particular weights - we are simply forcing the model to NOT rely on any particular weight too much. So, the result is, much like the weight normalization, that the values for weights will be regularized and will not become too large thereby reducing over-fitting.

**Results**

Now that your training is now complete, you should be able to see the training accuracy and the validation accuracy. The training accuracy keeps increasing as we train for more epochs and reaches a value that is consistently much higher than the validation accuracy. This is a clear case of over-fitting. The over-fitting problem is solved by using two regularization techniques.

## [Basic Image Classification with TensorFlow](https://github.com/shejz/Coursera-guided-projects/tree/master/Basic%20Image%20Classification%20with%20TensorFlow)
In this project, you will learn the basics of using Keras with TensorFlow as its backend and use it to solve a basic image classification problem. By the end of this project, you will have created, trained, and evaluated a Neural Network model that will be able to predict digits from hand-written images with a high degree of accuracy. 

## [Facial Expression Recognition with Keras](https://github.com/shejz/Coursera-guided-projects/tree/master/Facial%20Recognition%20with%20Keras)
In this project, you will build and train a convolutional neural network (CNN) in Keras from scratch to recognize facial expressions. The data consists of 48x48 pixel grayscale images of faces. The objective is to classify each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). You will use OpenCV to automatically detect faces in images and draw bounding boxes around them. Once you have trained, saved, and exported the CNN, you will directly serve the trained model to a web interface and perform real-time facial expression recognition on video and image data. 

**Learning Objectives**

- Develop a facial expression recognition model in Keras
- Build and train a convolutional neural network (CNN)
- Deploy the trained model to a web interface with Flask
- Apply the model to real-time video streams and image data

## [Create Custom Layers in Keras](https://github.com/shejz/Coursera-guided-projects/tree/master/Create%20Custom%20Layers%20in%20Keras)
In this project-based course, you will learn how to create a custom layer in Keras and create a model using the custom layer. We will create a simplified version of a Parametric ReLU layer and use it in a neural network model. Then we will use the neural network to solve a multi-class classiﬁcation problem. We will also compare our activation layer with the more commonly used ReLU activation layer.

**Learning Objectives**

- How to create custom layers in Keras.
- How to use custom layers in Keras models.

## [Creating Custom Callbacks in Keras](https://github.com/shejz/Coursera-guided-projects/tree/master/Creating%20Custom%20Callbacks%20in%20Keras)
In this project-based course, you will learn to create a custom callback function in Keras and use the callback during a model training process. We will implement the callback function to perform three tasks: Write a log file during the training process, plot the training metrics in a graph during the training process, and reduce the learning rate during the training with each epoch.

## [Build Multilayer Perceptron Models with Keras](https://github.com/shejz/Coursera-guided-projects/tree/master/Build%20Multilayer%20Perceptron%20Models%20with%20Keras)
In this project, you will build and train a multilayer perceptronl (MLP) model using Keras, with Tensorflow as its backend. We will be working with the Reuters dataset, a set of short newswires and their topics, published by Reuters in 1986. It's a very simple, widely used toy dataset for text classification. There are 46 different topics, some of which are more represented than others. But each topic has at least 10 examples in the training set. So in this project, you will build a MLP feed-forward neural network to classify Reuters newswires into 46 different mutually-exclusive topics.

## [Sentiment Analysis with Deep Learning using BERT](https://github.com/shejz/Coursera-guided-projects/tree/master/Sentiment%20Analysis%20with%20Deep%20Learning%20using%20BERT)
In this project, you will learn how to analyze a dataset for sentiment analysis. You will learn how to read in a PyTorch BERT model, and adjust the architecture for multi-class classification. You will learn how to adjust an optimizer and scheduler for ideal training and performance. In fine-tuning this model, you will learn how to design a train and evaluate loop to monitor model performance as it trains, including saving and loading models. 

## [Traffic Sign Classification](https://github.com/shejz/Coursera-guided-projects/tree/master/Traffic%20Sign%20Classification)
In this hands-on project, we will train deep learning models known as Convolutional Neural Networks (CNNs) to classify 43 traffic sign images. This project could be practically applied to self-driving cars. 

In this hands-on project we will go through the following tasks: 

1. Import libraries and datasets 
2. Images visualization
3. Convert images to gray-scale and perform normalization 
4. Build, compile and train deep learning model 
5. Assess trained model performance

**Learning Objectives**

- Understand the theory and intuition behind Convolutional Neural Networks (CNNs).
- Apply Python libraries to import and visualize dataset images.
- Perform image normalization and convert from color-scaled to gray-scaled images.
- Build a Convolutional Neural Network using Keras with Tensorflow 2.0 as a backend.
- Compile and fit Deep Learning model to training data.
- Assess the performance of trained CNN and ensure its generalization using various KPIs such as accuracy, precision and recall.
- Improve network performance using regularization techniques such as dropout.

## [Understanding Deepfakes with Keras](https://github.com/shejz/Coursera-guided-projects/tree/master/Understanding%20Deepfakes%20with%20Keras)
Implement DCGAN or Deep Convolutional Generative Adversarial Network, and you will train the network to generate realistic looking synthesized images. The term Deepfake is typically associated with synthetic data generated by Neural Networks which is similar to real-world, observed data - often with synthesized images, videos or audio. Through this hands-on project, we will go through the details of how such a network is structured, trained, and will ultimately generate synthetic images similar to hand-written digit 0 from the MNIST dataset.

## [Dimensionality Reduction using an Autoencoder](https://github.com/shejz/Coursera-guided-projects/tree/master/Dimensionality%20Reduction%20using%20an%20Autoencoder)
In this project, you will learn how to generate your own high-dimensional dummy dataset. You will then learn how to preprocess it effectively before training a baseline PCA model. You will learn the theory behind the autoencoder, and how to train one in scikit-learn. You will also learn how to extract the encoder portion of it to reduce dimensionality of your input data. In the course of this project, you will also be exposed to some basic clustering strength metrics.

## [Detecting COVID-19 with Chest X-Ray](https://github.com/shejz/Coursera-guided-projects/tree/master/Detecting%20COVID-19%20with%20Chest%20X-Ray%20using%20PyTorch)
Use a ResNet-18 model and train it on a COVID-19 Radiography dataset. This dataset has nearly 3000 Chest X-Ray scans which are categorized in three classes - Normal, Viral Pneumonia and COVID-19. Our objective in this tutorial is to create an image classification model that can predict Chest X-Ray scans that belong to one of the three classes with a reasonably high accuracy.

