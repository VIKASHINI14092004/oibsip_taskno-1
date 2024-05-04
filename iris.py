import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import Decision TreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt

# Load data
df = pd.read_csv('Iris.csv')

# Preprocess data
df.drop('Id', axis=1, inplace=True)
le = LabelEncoder()
df['Species'] = le.fit_transform(df['Species'])

# Split data
X = df.drop('Species', axis=1)
y = df['Species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = Decision TreeClassifier()
model.fit(X_train, y_train)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
cm = confusion_matrix(y_test, predictions)

# Print results
print("Accuracy:", accuracy)

# Plot confusion matrix
plt.figure(facecolor='white')
ax = plt.subplot()
cax = ax.matshow(cm, cmap=plt.cm.Blues)
plt.colorbar(cax)
ax.set_xlabel('Predicted labels')
ax.set_ylabel('True labels')
ax.set_xticklabels([''] + le.classes_.tolist())
ax.set_yticklabels([''] + le.classes_.tolist())
plt.title('Confusion Matrix')
plt.show()
