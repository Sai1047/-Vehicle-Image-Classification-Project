Vehicle Image Classification Report
### push

Introduction

Image classification has become an essential application of machine learning, enabling automation in fields such as transportation, security, and logistics. This project aims to classify images of vehicles into predefined categories: bus, car, motorcycle, and truck. By leveraging conventional machine learning and deep learning (CNN), we demonstrate the ability to train a model that can accurately classify vehicle types from images. Such a system has practical implications in traffic monitoring, autonomous driving, and fleet management.

Dataset Description

The dataset used for this project contains vehicle images categorized into four classes: Bus, Car, Motorcycle, and Truck. It was sourced from Kaggle and contains a structured format where each class has its own folder. The dataset was preprocessed to resize all images to 128x128 pixels for uniformity. Additionally, data augmentation techniques such as rotation, flipping, and scaling were applied to enhance model robustness and mitigate overfitting.

- **Number of Classes**: 4
- **Image Dimensions**: 128x128 pixels
- **Augmentation Techniques**: Rotation, flipping, scaling
- **Train-Validation Split**: 80%-20%

Methodology

The project involved the following steps:

1. **Preprocessing**: The images were resized to a uniform size and normalized to ensure pixel values ranged between 0 and 1. Data augmentation was employed to artificially increase the size and variability of the dataset.

2. **Model Training**:

   - **Conventional Machine Learning**: Handcrafted features such as color histograms and texture descriptors were extracted and used with classifiers like Support Vector Machines (SVM) and Random Forests.
   - **Convolutional Neural Network (CNN)**: A deep learning model was designed with three convolutional layers followed by max-pooling layers. Dropout layers were added to prevent overfitting, and the final dense layer predicted the class probabilities.

3. **Evaluation**: The model's performance was evaluated using accuracy, precision, recall, and a confusion matrix on the validation set.

Results and Discussion

- **Conventional Techniques**: Achieved moderate accuracy (\~70%) due to the limited ability to extract complex features from images.
- **CNN Model**:
  - **Training Accuracy**: \~95%
  - **Validation Accuracy**: \~90%

The CNN model outperformed conventional techniques, demonstrating its ability to learn hierarchical features directly from images. However, some misclassifications occurred in cases of visually similar classes (e.g., buses and trucks). The use of data augmentation significantly improved the model's generalization capabilities.

The confusion matrix revealed that the motorcycle class was most accurately predicted, while the bus class occasionally overlapped with the truck class due to similar shapes and features.

Conclusion

This project successfully demonstrated the use of CNNs for vehicle image classification. The model achieved high accuracy, proving its potential for real-world applications such as traffic monitoring and vehicle recognition. Future work could focus on expanding the dataset, fine-tuning the model, and integrating real-time classification capabilities for practical deployment.

