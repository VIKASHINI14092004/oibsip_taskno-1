# oibsip_taskno1 : Iris Flower Classification Project
Overview:
This project aims to classify Iris flowers into three species (setosa, versicolor, and virginica) based on their sepal and petal dimensions. The classification is performed using a Decision Tree Classifier implemented in Python with the help of the scikit-learn library.

Dataset:
The dataset used in this project is the famous Iris dataset, which is often used for classification tasks in machine learning. The dataset contains 150 samples of Iris flowers, with four features each: sepal length, sepal width, petal length, and petal width. Each sample is labeled with the species of Iris it belongs to.

Methodology:
Data Loading and Preprocessing: The dataset is loaded from a CSV file ('Iris.csv') using the Pandas library. The 'Id' column is dropped as it is not relevant to the classification task. The species names are encoded into numerical labels using the LabelEncoder from scikit-learn.
Data Splitting: The dataset is split into training and testing sets with a ratio of 80:20, where 80% of the data is used for training the model and 20% for evaluating its performance.
Model Training: A Decision Tree Classifier is chosen as the model for classification. The model is trained on the training data using the fit() method.
Model Evaluation: The trained model is evaluated on the testing data to assess its performance. The accuracy of the model is calculated using the accuracy_score() function, and a confusion matrix is generated using the confusion_matrix() function.
Results:
The accuracy of the trained model on the testing data is printed, and a confusion matrix is plotted to visualize the performance of the model.


Accuracy: 1.0

Conclusion:
The Decision Tree Classifier achieves perfect accuracy (1.0) on the testing data, indicating that it can effectively classify Iris flowers into their respective species based on their sepal and petal dimensions.

