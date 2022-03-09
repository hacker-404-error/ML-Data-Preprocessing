# Data Preprocessing in Machine learning
Data preprocessing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model.

## Why do we need Data Preprocessing?
A real-world data generally contains noises, missing values, and maybe in an unusable format which cannot be directly used for machine learning models. Data preprocessing is required tasks for cleaning the data and making it suitable for a machine learning model which also increases the accuracy and efficiency of a machine learning model.

## It involves below steps:
- `Getting the dataset`
- `Importing libraries`
- `Importing datasets`
- `Finding Missing Data`
- `Encoding Categorical Data`
- `Splitting dataset into training and test set`
- `Feature scaling`

## 1) Get the Dataset
To create a machine learning model, the first thing we required is a dataset as a machine learning model completely works on data. The collected data for a particular problem in a proper format is known as the dataset.

[Click Here](https://github.com/hacker-404-error/Machine_Learning_Fundamentals#popular-sources-for-machine-learning-datasets)  for All sources of datasets

## 2) Importing Libraries
There are three specific libraries that we will use for data preprocessing, which are:

- **Numpy** : Numpy Python library is used for including any type of mathematical operation in the code. It is the fundamental package for scientific calculation in Python. It also supports to add large, multidimensional arrays and matrices. So, in Python, we can import it as:
```
import numpy as np 
``` 

- **Matplotlib** : The second library is matplotlib, which is a Python 2D plotting library, and with this library, we need to import a sub-library pyplot. This library is used to plot any type of charts in Python for the code. It will be imported as below:
```
import matplotlib.pyplot as mpt
```

- **Pandas** : The last library is the Pandas library, which is one of the most famous Python libraries and used for importing and managing the datasets. It is an open-source data manipulation and analysis library
```
import pandas as pd
```

## 3) Importing the Datasets
to import the dataset, we will use read_csv() function of pandas library, which is used to read a csv file and performs various operations on it.
```
data_set= pd.read_csv('Dataset.csv')  
``` 
Here in this Project we are going to use this Dataset:

|     | Country | Age  | Salary  | Purchased |
| --- | ------- | ---- | ------- | --------- |
| 0   | France  | 44.0 | 72000.0 | No        |
| 1   | Spain   | 27.0 | 48000.0 | Yes       |
| 2   | Germany | 30.0 | 54000.0 | No        |
| 3   | Spain   | 38.0 | 61000.0 | No        |
| 4   | Germany | 40.0 | NaN     | Yes       |
| 5   | France  | 35.0 | 58000.0 | Yes       |
| 6   | Spain   | NaN  | 52000.0 | No        |
| 7   | France  | 48.0 | 79000.0 | Yes       |
| 8   | Germany | 50.0 | 83000.0 | No        |
| 9   | France  | 37.0 | 67000.0 | Yes       |

## 4) Handling Missing data:
If our dataset contains some missing data, then it may create a huge problem for our machine learning model.

Ways to handle missing data:

- `By deleting the particular row` : In this way, we just delete the specific row or column which consists of null values. But this way is not so efficient and removing data may lead to loss of information which will not give the accurate output.

- `By calculating the mean` : In this way, we will calculate the mean of that column or row which contains any missing value and will put it on the place of missing value. This strategy is useful for the features which have numeric data such as age, salary, year, etc. Here, we will use this approach.

To handle missing values, we will use Scikit-learn library in our code, which contains various libraries for building machine learning models. Here we will use Imputer class of sklearn.
```
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

```

## 5) Encoding Categorical data:
Since machine learning model completely works on mathematics and numbers, but if our dataset would have a categorical variable, then it may create trouble while building the model. So it is necessary to encode these categorical variables into numbers.
There are many ways to Encode Categorical data like `Label Encoding` , `Dummy Encoding` etc.

As we have '`Country`' and '`Purchased`' as a Categorical data we must apply Encoding on these data only.

### i] For Country Variable:

**`NOTE`**
```
- If we use Label Encoding in 'Country' Data There Might be Some serious Problem.
- there are three country variables, and these variables are encoded into 0, 1, and 2.
- By these values, the machine learning model may assume that there is some correlation
  between these variables which will produce the wrong output. 
- So to remove this issue, we will use dummy encoding.
``` 
#### Dummy Encoding:
`Dummy variables` are those variables which have values 0 or 1. The 1 value gives the presence of that variable in a particular column, and rest variables become 0. With dummy encoding, we will have a number of columns equal to the number of categories.

In our dataset, we have 3 categories so it will produce three columns having 0 and 1 values. For Dummy Encoding, we will use **OneHotEncoder** class of preprocessing library.

```
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```
### ii] For Purchased Variable:

we will only use `labelencoder` object of LableEncoder class. Here we are not using OneHotEncoder class because the purchased variable has only two categories yes or no, and which are automatically encoded into 0 and 1.

```
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)
```

## 6) Splitting the Dataset into the Training set and Test set
In machine learning data preprocessing, we divide our dataset into a training set and test set. This is one of the crucial steps of data preprocessing as by doing this, we can enhance the performance of our machine learning model.

<br>
<div align="center">
    <img src="./imgs/train%20and%20test.jpg" alt="Train and Test">
</div>
<br>

- *`Training Set`*: A subset of dataset to train the machine learning model, and we already know the output.

- *`Test set`*: A subset of dataset to test the machine learning model, and by using the test set, model predicts the output.

For splitting the dataset, we will use Scikit-learn library in our code

Data set is splited randomly with 80% data on Training Set 20% on Test Set.
```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 1)
```
- `x_train`: features for the training data
- `x_test `: features for testing data
- `y_train`: Dependent variables for training data
- `y_test `: Independent variable for testing data

## 7) Feature Scaling

- Feature scaling is the final step of data preprocessing in machine learning. It is a technique to standardize the independent variables of the dataset in a specific range. In feature scaling, we put our variables in the same range and in the same scale so that no any variable dominate the other variable.

- As we can see, the age and salary column values are not on the same scale. A machine learning model is based on Euclidean distance, and if we do not scale the variable, then it will cause some issue in our machine learning model.

<br>
<div align="center">
    <img src="https://static.javatpoint.com/tutorial/machine-learning/images/data-preprocessing-machine-learning-8.png" alt="Euclidean distance">
</div>
<br>

- If we compute any two values from age and salary, then salary values will dominate the age values, and it will produce an incorrect result. So to remove this issue, we need to perform feature scaling for machine learning.

**There are two ways to perform feature scaling in machine learning:**

- ###  `Standardization`
<br>
<div align="center">
    <img src="https://static.javatpoint.com/tutorial/machine-learning/images/data-preprocessing-machine-learning-9.png" alt="Standardization">
</div>
<br>

- ### `Normalization`
<br>
<div align="center">
    <img src="https://static.javatpoint.com/tutorial/machine-learning/images/data-preprocessing-machine-learning-10.png" alt="Normalization">
</div>
<br>

Here, we will use the standardization method for our dataset.
For feature scaling, we will import StandardScaler class of sklearn.preprocessing library as:

```
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_test[:, 3:] = sc.transform(X_test[:, 3:])
```

## Combining all the steps:
Now, in the end, we can combine all the steps together to make our complete code more understandable.

[Complete Code](https://github.com/hacker-404-error/ML-Data-Preprocessing/blob/master/data_preprocessing_tools.ipynb)

If You want To Run The Code Then You Can Use Google Colab [![Open In Collab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hacker-404-error/ML-Data-Preprocessing/blob/master/data_preprocessing_tools.ipynb)

*`NOTE`* : Before running the Program upload [This](https://github.com/hacker-404-error/ML-Data-Preprocessing/blob/master/Data.csv) Dataset.

------
### Previous Topics : 
- [Fundamentals Of Machine Learning](https://github.com/hacker-404-error/Machine_Learning_Fundamentals)
------

### Next Topics : 
- [Linear Regression]()
- [Classification]()
--------

### Created And Coded By:
<a href = "https://github.com/hacker-404-error">Pritam Das</a>

<a href="https://github.com/hacker-404-error"><img src="https://i.ibb.co/yYd2Xjb/In-Shot-20220309-143908060.png" alt="Avatar" style="border-radius: 50%; width:70px"></a>



## 🔗 Feedback 
If you have any feedback, please reach out to me at [![linkedin](https://img.shields.io/badge/linkedin-0A66C2?style=for-the-badge&logo=linkedin&logoColor=white)](https://www.linkedin.com/in/pritam-das-7489ab223/)
