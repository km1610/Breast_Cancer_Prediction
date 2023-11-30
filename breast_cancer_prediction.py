import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#load data on dataframe
df = pd.read_csv('Breast Cancer Prediction from Dataset\data.csv')

#display dataframe
df.head()

#count of rows and columns
df.shape

#count number of null(empty) values
df.isna().sum()

# Drop the column with null values
df.dropna(axis=1,inplace=True)

#Get count of number of M or B cells in diagnosis
df['diagnosis'].value_counts()

#Get Datatypes of each column in our dataset
df.dtypes

#Encode the diagnosis values
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.iloc[:,1]=le.fit_transform(df.iloc[:,1].values)

#Splitting the dataset into independent and dependent datasets
x=df.iloc[:,2:].values
y=df.iloc[:,1].values

#Splitting datasets into training(75%) and testing(25%)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.25)

#Scaling the data(feature scaling)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.fit_transform(x_test)

#build a logistic regression classifier
from sklearn.linear_model import LogisticRegression
c = LogisticRegression()
c.fit(x_train,y_train)

#make use of trained model to make predictions on test data
p = c.predict(x_test)

#plot confusion matrix
from sklearn.metrics import confusion_matrix
import seaborn as sns
cm = confusion_matrix(y_test,p)
sns.heatmap(cm,annot=True)
plt.show()

#get accuracy score for model
from sklearn.metrics  import accuracy_score
print(accuracy_score(y_test,p))


from sklearn import tree
from sklearn.model_selection import train_test_split

clf = tree.DecisionTreeClassifier(random_state=42)

# Train the classifier on the training set
clf.fit(x_train, y_train)

y_pred = clf.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.ensemble import RandomForestClassifier

# Create a Random Forest classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the classifier on the training set
rf_classifier.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = rf_classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Create and train the linear regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")

from sklearn import svm

# Create a SVM classifier
svm_classifier = svm.SVC(kernel='rbf')  # You can choose different kernels like 'linear', 'rbf', 'poly', etc.

# Train the classifier on the training set
svm_classifier.fit(x_train, y_train)

# Make predictions on the testing set
y_pred = svm_classifier.predict(x_test)
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True)
# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

