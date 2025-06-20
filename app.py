import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv('forms/anemia.csv')
print(df.head())
print(df.info())
print(df.shape)
print(df.isnull().sum())
results=df['Result'].value_counts()
results.plot(kind='bar',color=['blue','green'])
plt.xlabel('Result')
plt.ylabel('Frequency')
plt.title('Count of Result')
plt.show()
from sklearn.utils import resample
import pandas as pd

# Separate majority and minority classes
majorclass = df[df['Result'] == 0]
minorclass = df[df['Result'] == 1]

# Undersample the majority class to match the minority class size
major_downsample = resample(
    majorclass,
    replace=False,                     # sample without replacement
    n_samples=len(minorclass),         # match minority class count
    random_state=42                    # reproducibility
)

# Combine downsampled majority class with minority class
df = pd.concat([major_downsample, minorclass])

# Check the class distribution
print(df['Result'].value_counts())
results_balanced =df['Result'].value_counts()
results_balanced.plot(kind='bar',color=['blue','green'])
plt.xlabel('Result')
plt.ylabel('Frequency')
plt.title('Count of Result')
plt.show()
print(df.describe())
results=df['Gender'].value_counts()
results.plot(kind='bar',color=['orange','green'])
plt.xlabel('Gender')
plt.ylabel('Frequency')
plt.title('Count of Result')
plt.show()
sns.histplot(df['Hemoglobin'],kde=True, stat='count')
plt.title("Hemoglobin Level Distribution")
plt.xlabel("Hemoglobin")
plt.ylabel("Count")
plt.show()
plt.figure(figsize=(6,6))
ax = sns.barplot(y=df['Hemoglobin'], x=df['Gender'], hue=df['Result'], ci=None)

# Set x-axis labels
ax.set(xlabel=['male', 'female'])

# Add value labels on the bars
ax.bar_label(ax.containers[0])
ax.bar_label(ax.containers[1])

# Set plot title
plt.title("Mean Hemoglobin by Gender and Result")

# Show the plot
plt.show()

sns.pairplot(df)
plt.show()

sns.heatmap(df.corr(), annot=True, cmap='RdYlGn', linewidths=0.2)
fig = plt.gcf()
fig.set_size_inches(10, 8)
plt.show()

X=df.drop('Result',axis=1)
print(X)

Y=df['Result']
print(Y)

from sklearn.model_selection import train_test_split

# Split the dataset into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=20)

# Print the shapes of the resulting datasets
print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

logistic_regression = LogisticRegression()
logistic_regression.fit(x_train, y_train)
y_pred = logistic_regression.predict(x_test)

acc_lr = accuracy_score(y_test, y_pred)
c_lr = classification_report(y_test, y_pred)

print('Accuracy Score: ', acc_lr)
print(c_lr)

from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier()
random_forest.fit(x_train, y_train)
y_pred = random_forest.predict(x_test)

acc_rf = accuracy_score(y_test, y_pred)
c_rf = classification_report(y_test, y_pred)

print('Accuracy Score: ', acc_rf)
print(c_rf)

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

decision_tree_model = DecisionTreeClassifier()
decision_tree_model.fit(x_train, y_train)
y_pred = decision_tree_model.predict(x_test)

acc_dt = accuracy_score(y_test, y_pred)
c_dt = classification_report(y_test, y_pred)

print('Accuracy Score: ', acc_dt)
print(c_dt)

from sklearn.naive_bayes import GaussianNB

NB = GaussianNB()
NB.fit(x_train, y_train)
y_pred = NB.predict(x_test)

acc_nb = accuracy_score(y_test, y_pred)
c_nb = classification_report(y_test, y_pred)

print('Accuracy Score: ', acc_nb)
print(c_nb)

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

support_vector = SVC()
support_vector.fit(x_train, y_train)
y_pred = support_vector.predict(x_test)

acc_svc = accuracy_score(y_test, y_pred)
c_svc = classification_report(y_test, y_pred)

print('Accuracy Score: ', acc_svc)
print(c_svc)

from sklearn.ensemble import GradientBoostingClassifier

GBC = GradientBoostingClassifier()
GBC.fit(x_train, y_train)
y_pred = GBC.predict(x_test)

acc_gbc = accuracy_score(y_test, y_pred)
c_gbc = classification_report(y_test, y_pred)

print('Accuracy Score: ', acc_gbc)
print(c_gbc)

prediction = GBC.predict([[0,11.6,22.3,30.9,74.5]])
prediction[0]
if prediction[0]==0:
    print("The person is not affected by anemia")
elif prediction[0]==1:
    print("The person is affected by anemia")  

    model = pd.DataFrame({'Model':['Linear Regression','Decision Tree Classifier','RandomForest Classifier',
                               'Gaussian Naive Bayes','Support Vector Classifier', 'Gradient Boost Classifier'],
                      'Score':[acc_lr,acc_dt,acc_rf,acc_nb,acc_svc,acc_gbc],
                     })
    print(model)

import pickle
import warnings
pickle.dump(GBC,open('model.pkl','wb'))

import numpy as np
import pickle
import pandas as pd
from flask import Flask, request, render_template
app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))
@app.route('/')  # Route to display the home page
def home():
    return render_template('index.html')

@app.route('/predict', methods=["POST"])
def predict():
    Gender = float(request.form["Gender"])
    Hemoglobin = float(request.form["Hemoglobin"])
    MCH = float(request.form["MCH"])
    MCHC = float(request.form["MCHC"])
    MCV = float(request.form["MCV"])

    features_values = np.array([[Gender, Hemoglobin, MCH, MCHC, MCV]])  # reshape to 2D array

    df = pd.DataFrame(features_values, columns=['Gender', 'Hemoglobin', 'MCH', 'MCHC', 'MCV'])
    print(df)

    prediction = model.predict(df)
    print(prediction[0])
    result = prediction[0]

    if prediction[0] == 0:
        result = "You don't have any Anemic Disease"
    elif prediction[0] == 1:
        result = "You have anemic disease"

    text = "Hence, based on calculation: "
    return render_template("predict.html", prediction_text=text + str(result))

if __name__ == "__main__":
    app.run(debug=True)