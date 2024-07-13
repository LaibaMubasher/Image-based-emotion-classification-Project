# Image-based-emotion-classification-Project

#Overview
This project focuses on understanding and classifying human emotions using images. Emotions such as pain, happiness, sadness, anger, disgust, and fear play vital roles in our interactions. By analyzing visual cues like facial expressions and body language, we aim to infer emotional states from images. Our goal is to leverage computer vision and machine learning advancements to create empathetic and intuitive technologies that can understand human emotions.

#Motivation
Our objective is to deepen our understanding of image-based emotion classification using a Kaggle dataset of images depicting six fundamental emotions. This dataset provides rich visual cues for analysis, enabling us to explore feature extraction, classification algorithms, and model evaluation in computer vision.

#Dataset Description
The dataset, sourced from Kaggle, consists of 1200 images evenly distributed across six emotions:

Happiness: 230 images
Anger: 214 images
Fear: 163 images
Disgust: 201 images
Pain: 168 images
Sadness: 224 images
The images are uniform in size, ensuring consistent analysis and effective model application.

#Data Visualization
We utilized histograms, line graphs, bar graphs, pie charts, and scatter plots to gain insights into the data. For instance, the bar graph clearly illustrates the number of images in each emotion class, aiding in data understanding and communication.

#Model Development
#Convolutional Neural Network (CNN)
#Architecture:
Convolutional layers: 32, 64, 128 filters with ReLU activation and MaxPooling.
Flatten layer.
Dense layers: 512 neurons with ReLU, followed by a softmax layer for classification.
#Training:
Used 80% of the data for training and 20% for testing.
Applied data augmentation to increase dataset diversity.

#Multi-Layer Perceptron (MLP)
#Architecture:
Sequential model with Flatten, Dense, and Dropout layers.
#Training:
Applied data augmentation and hyperparameter tuning.

#Transfer Learning with VGG19
#Architecture:
VGG19 base model with additional Dense and Dropout layers.
Training:
Increasing the dropout rate helped reduce overfitting.
#Transfer Learning with Inception V3
#Architecture:
Inception V3 base model with additional Dense and Dropout layers.
Training:
Reducing model complexity and increasing dropout rate improved accuracy to 0.62.
Conclusion
The project explored various models for emotion classification from images. While CNN and MLP models showed potential, transfer learning models like VGG19 and Inception V3 demonstrated higher accuracy and robustness. Techniques such as data augmentation and dropout were essential in improving model performance and reducing overfitting. Our findings suggest that advanced pre-trained models are more effective for this task, especially when fine-tuned with appropriate hyperparameters.
