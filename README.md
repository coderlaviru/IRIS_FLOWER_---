# IRIS_FLOWER_---
This repository contains the code and data for the Irish Flower Model, a machine learning project aimed at classifying different species of Irish flowers based on their features. The model is built using Python and various machine learning libraries.
# 1. Know the data
# Importing Libraries


```python
import numpy as np
import pandas as pd

#Importing tools for visualization 
import matplotlib.pyplot as plt 
import seaborn as sns 
#Import evaluation metric librarie s
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, classification_report 
from sklearn.preprocessing import LabelEncoder
#Libraries used for data  prprocessing 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import RepeatedStratifiedKFold

#Library used for ML Model implementation
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier 
from sklearn.naive_bayes import GaussianNB
import xgboost as xgb #Xtreme Gradient Boosting 
#librries used for ignore warnings 
import warnings
warnings.filterwarnings('ignore')
%matplotlib inline
```

# Dataset Loading


```python
iris=pd.read_csv(r"C:\Users\lakshita\Desktop\datasets\iris.csv")
```

# Dataset First View


```python
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>145</th>
      <td>146</td>
      <td>6.7</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>146</th>
      <td>147</td>
      <td>6.3</td>
      <td>2.5</td>
      <td>5.0</td>
      <td>1.9</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>147</th>
      <td>148</td>
      <td>6.5</td>
      <td>3.0</td>
      <td>5.2</td>
      <td>2.0</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>148</th>
      <td>149</td>
      <td>6.2</td>
      <td>3.4</td>
      <td>5.4</td>
      <td>2.3</td>
      <td>Iris-virginica</td>
    </tr>
    <tr>
      <th>149</th>
      <td>150</td>
      <td>5.9</td>
      <td>3.0</td>
      <td>5.1</td>
      <td>1.8</td>
      <td>Iris-virginica</td>
    </tr>
  </tbody>
</table>
</div>




```python
iris.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
      <td>150.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>75.500000</td>
      <td>5.843333</td>
      <td>3.054000</td>
      <td>3.758667</td>
      <td>1.198667</td>
    </tr>
    <tr>
      <th>std</th>
      <td>43.445368</td>
      <td>0.828066</td>
      <td>0.433594</td>
      <td>1.764420</td>
      <td>0.763161</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.000000</td>
      <td>4.300000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>0.100000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>38.250000</td>
      <td>5.100000</td>
      <td>2.800000</td>
      <td>1.600000</td>
      <td>0.300000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>75.500000</td>
      <td>5.800000</td>
      <td>3.000000</td>
      <td>4.350000</td>
      <td>1.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>112.750000</td>
      <td>6.400000</td>
      <td>3.300000</td>
      <td>5.100000</td>
      <td>1.800000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>150.000000</td>
      <td>7.900000</td>
      <td>4.400000</td>
      <td>6.900000</td>
      <td>2.500000</td>
    </tr>
  </tbody>
</table>
</div>



# Dataset Rows & Columns count


```python
# Correcting the use of iris.info()
print(iris.info())
print("Number of rows ", iris.shape[0])
print("Number of Columns ",iris.shape[1])
print(iris.head(150))
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             150 non-null    int64  
     1   SepalLengthCm  150 non-null    float64
     2   SepalWidthCm   150 non-null    float64
     3   PetalLengthCm  150 non-null    float64
     4   PetalWidthCm   150 non-null    float64
     5   Species        150 non-null    object 
    dtypes: float64(4), int64(1), object(1)
    memory usage: 7.2+ KB
    None
    Number of rows  150
    Number of Columns  6
          Id  SepalLengthCm  SepalWidthCm  PetalLengthCm  PetalWidthCm  \
    0      1            5.1           3.5            1.4           0.2   
    1      2            4.9           3.0            1.4           0.2   
    2      3            4.7           3.2            1.3           0.2   
    3      4            4.6           3.1            1.5           0.2   
    4      5            5.0           3.6            1.4           0.2   
    ..   ...            ...           ...            ...           ...   
    145  146            6.7           3.0            5.2           2.3   
    146  147            6.3           2.5            5.0           1.9   
    147  148            6.5           3.0            5.2           2.0   
    148  149            6.2           3.4            5.4           2.3   
    149  150            5.9           3.0            5.1           1.8   
    
                Species  
    0       Iris-setosa  
    1       Iris-setosa  
    2       Iris-setosa  
    3       Iris-setosa  
    4       Iris-setosa  
    ..              ...  
    145  Iris-virginica  
    146  Iris-virginica  
    147  Iris-virginica  
    148  Iris-virginica  
    149  Iris-virginica  
    
    [150 rows x 6 columns]
    

# Dataset Information


```python
# Checking information about the dataset using info
iris.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 150 entries, 0 to 149
    Data columns (total 6 columns):
     #   Column         Non-Null Count  Dtype  
    ---  ------         --------------  -----  
     0   Id             150 non-null    int64  
     1   SepalLengthCm  150 non-null    float64
     2   SepalWidthCm   150 non-null    float64
     3   PetalLengthCm  150 non-null    float64
     4   PetalWidthCm   150 non-null    float64
     5   Species        150 non-null    object 
    dtypes: float64(4), int64(1), object(1)
    memory usage: 7.2+ KB
    

# Duplicate Values


```python
dup = iris.duplicated().sum()
print(f'Number of duplicate rows: {dup}')
```

    Number of duplicate rows: 0
    

# Dropping duplicate rows


```python
# Dropping duplicate rows
iris=iris.drop_duplicates()
```

# after dropping duplicates


```python
# Checking the number of rows again to see if duplicates were dropped
print(f'Number of rows after dropping duplicates: {iris.shape[0]}')
```

    Number of rows after dropping duplicates: 150
    

# Count plot for the species distribution


```python
# Count plot for the species distribution
sns.countplot(data= iris, x ='SepalLengthCm')
plt.title('Distribution of Iris Species')
plt.show()
```


    
![png](output_19_0.png)
    



```python
# Scatter plot for Sepal Length vs Sepal Width
sns.scatterplot(data= iris, x='SepalLengthCm', y='SepalWidthCm', hue='Species')
plt.title('SepalLengthCm vs. SepalWidthCm')
plt.show()
```


    
![png](output_20_0.png)
    



```python
# Pair plot for visualizing relationships between all features
sns.pairplot(iris, hue='Species')
plt.show()
```


    
![png](output_21_0.png)
    



```python
#checking number of rows and column of the dataset using shape
print("Number of rows: ",iris.shape[0])
print("Number of columns: ",iris.shape[1])
```

    Number of rows:  150
    Number of columns:  6
    

# Missing Values/Null Values


```python
iris.isnull().sum()
```




    Id               0
    SepalLengthCm    0
    SepalWidthCm     0
    PetalLengthCm    0
    PetalWidthCm     0
    Species          0
    dtype: int64



# 2. Understanding The Variables


```python
# Dataset Columns
iris.columns
```




    Index(['Id', 'SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm',
           'Species'],
          dtype='object')




```python
iris.describe (include='all').round(2)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Id</th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>150.00</td>
      <td>150.00</td>
      <td>150.00</td>
      <td>150.00</td>
      <td>150.00</td>
      <td>150</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>3</td>
    </tr>
    <tr>
      <th>top</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>50</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>75.50</td>
      <td>5.84</td>
      <td>3.05</td>
      <td>3.76</td>
      <td>1.20</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>std</th>
      <td>43.45</td>
      <td>0.83</td>
      <td>0.43</td>
      <td>1.76</td>
      <td>0.76</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>min</th>
      <td>1.00</td>
      <td>4.30</td>
      <td>2.00</td>
      <td>1.00</td>
      <td>0.10</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>38.25</td>
      <td>5.10</td>
      <td>2.80</td>
      <td>1.60</td>
      <td>0.30</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>75.50</td>
      <td>5.80</td>
      <td>3.00</td>
      <td>4.35</td>
      <td>1.30</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>112.75</td>
      <td>6.40</td>
      <td>3.30</td>
      <td>5.10</td>
      <td>1.80</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>max</th>
      <td>150.00</td>
      <td>7.90</td>
      <td>4.40</td>
      <td>6.90</td>
      <td>2.50</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



# Check Unique Values for each variable.


```python
# Check Unique Values for each variable.
for i in iris.columns.tolist():
  print("No. of unique values in",i,"is",iris[i].nunique())
```

    No. of unique values in Id is 150
    No. of unique values in SepalLengthCm is 35
    No. of unique values in SepalWidthCm is 23
    No. of unique values in PetalLengthCm is 43
    No. of unique values in PetalWidthCm is 22
    No. of unique values in Species is 3
    

# 3. Data Wrangling

# Data Wrangling Code


```python
# We don't need the 1st column so let's drop that
iris=iris.iloc[:,1:]
```


```python
# New updated dataset
iris.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>SepalLengthCm</th>
      <th>SepalWidthCm</th>
      <th>PetalLengthCm</th>
      <th>PetalWidthCm</th>
      <th>Species</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5.1</td>
      <td>3.5</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>1</th>
      <td>4.9</td>
      <td>3.0</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.7</td>
      <td>3.2</td>
      <td>1.3</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4.6</td>
      <td>3.1</td>
      <td>1.5</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5.0</td>
      <td>3.6</td>
      <td>1.4</td>
      <td>0.2</td>
      <td>Iris-setosa</td>
    </tr>
  </tbody>
</table>
</div>



# 4. Data Vizualization, Storytelling & Experimenting with charts : Understand the relationships between variables

# Chart - 1 : Distribution of Numerical Variables


```python
#chart 1 histogram visual code for distribution of numerical variables
#create a figure with subject
plt.figure(figsize=(8,6))
plt.suptitle('distribution of Iris flower measurement',fontsize=14)

#create a 2x2 grid of subplot
plt.subplot(2,2,1) #subject 1 (top-left)
plt.hist(iris['SepalLengthCm'])
plt.title('Sepal Lenght Distribution')

plt.subplot(2,2,2) #subject 2 (top-right)
plt.hist(iris['SepalWidthCm'])
plt.title('sepal width distribution')

plt.subplot(2,2,3) #subject 3 (bottom-left)
plt.hist(iris['PetalLengthCm'])
plt.title('Petal lenght distribution')

plt.subplot(2,2,4) #subject 4 (bottom-right)
plt.hist(iris['PetalWidthCm'])
plt.title('Petal width distribution')

#display the subjects
plt.tight_layout() #help in adjusting the layout
plt.show()
```


    
![png](output_36_0.png)
    


# Chart - 2 Scatter plot visualization code for Sepal Length vs Sepal Width.


```python
# Define colors for each species and the corresponding species labels.
colors = ['red', 'yellow', 'green']
species = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
```


```python
# Create a scatter plot for Sepal Length vs Sepal Width for each species
for i in range(3):
    # Select data for the current species
    x = iris[iris['Species'] == species[i]]

    # Create a scatter plot with the specified color and label for the current species.
    plt.scatter(x['SepalLengthCm'], x['SepalWidthCm'], c=colors[i], label=species[i])

# Add labels to the x and y axes
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')

# Add a legend to identify species based on colors
plt.legend()

# Display the scatter plot
plt.show()
```


    
![png](output_39_0.png)
    


# Chart - 3 Scatter plot visualization code for Petal Length vs Petal Width.


```python
# Create a scatter plot for Petal Length vs Petal Width for each species.
for i in range(3):
    # Select data for the current species.
    x = iris[iris['Species'] == species[i]]

    # Create a scatter plot with the specified color and label for the current species.
    plt.scatter(x['PetalLengthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])

# Add labels to the x and y axes.
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')

# Add a legend to identify species based on colors.
plt.legend()

# Display the scatter plot.
plt.show()
```


    
![png](output_41_0.png)
    


# Chart - 4 Scatter plot visualization code for Sepal Length vs Petal Length.


```python
# Create a scatter plot for Sepal Length vs Petal Length for each species.
for i in range(3):
    # Select data for the current species.
    x = iris[iris['Species'] == species[i]]

    # Create a scatter plot with the specified color and label for the current species.
    plt.scatter(x['SepalLengthCm'], x['PetalLengthCm'], c=colors[i], label=species[i])

# Add labels to the x and y axes.
plt.xlabel('Sepal Length')
plt.ylabel('Petal Length')

# Add a legend to identify species based on colors.
plt.legend()

# Display the scatter plot.
plt.show()
```


    
![png](output_43_0.png)
    


# Chart - 5 : Sepal Width vs Petal Width


```python
# Chart - 5 Scatter plot visualization code for Sepal Width vs Petal Width.
# Create a scatter plot for Sepal Width vs Petal Width for each species.
for i in range(3):
    # Select data for the current species.
    x = iris[iris['Species'] == species[i]]

    # Create a scatter plot with the specified color and label for the current species.
    plt.scatter(x['SepalWidthCm'], x['PetalWidthCm'], c=colors[i], label=species[i])

# Add labels to the x and y axes.
plt.xlabel('Sepal Width')
plt.ylabel('Petal Width')

# Add a legend to identify species based on colors.
plt.legend()

# Display the scatter plot.
plt.show()
```


    
![png](output_45_0.png)
    


# Chart - 6 : Correlation Heatmap


```python
# Correlation Heatmap Visualization Code
corr_matrix = iris.corr()

# Plot Heatmap
plt.figure(figsize=(8, 4))
sns.heatmap(corr_matrix, annot=True, cmap='Reds_r')

# Setting Labels
plt.title('Correlation Matrix Heatmap')

# Display Chart
plt.show()
```


    
![png](output_47_0.png)
    


# 5. Feature Engineering & Data Pre-processing

# Categorical Encoding


```python
#encode the categorical columns
# create a labelEncoder object 
le=LabelEncoder()

#encode the "Species" column to convert the species names to numerical names to numerical labels
iris['Species']=le.fit_transform (iris['Species'])

#check the unique values in the Sepcies' column after encoding 
unique_species=iris['Species'].unique()

#display the unique encoded values
print("Encoded Species Value:")
print(unique_species) #'Iris-setosa'==0 ,'Iris-versicolor'==1  ,'Iris-virginica'==2
```

    Encoded Species Value:
    [0 1 2]
    

# Data Scaling


```python
# Defining the X and y
x=iris.drop(columns=['Species'], axis=1)
y=iris['Species']
```

# Data splitting


```python
# splitting the data to train and test
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3)
```


```python
# Checking the train distribution of dependent variable
y_train.value_counts()
```




    Species
    1    37
    0    34
    2    34
    Name: count, dtype: int64



# 6. ML Model Implementation


```python
def evaluate_model(model, x_train, x_test, y_train, y_test):
    '''The function will take model, x train, x test, y train, y test
    and then it will fit the model, then make predictions on the trained model,
    it will then print roc-auc score of train and test, then plot the roc, auc curve,
    print confusion matrix for train and test, then print classification report for train and test,
    then plot the feature importances if the model has feature importances,
    and finally it will return the following scores as a list:
    recall_train, recall_test, acc_train, acc_test, F1_train, F1_test
    '''

    # Fit the model to the training data.
    model.fit(x_train, y_train)

    # make predictions on the test data
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)

    # calculate confusion matrix
    cm_train = confusion_matrix(y_train, y_pred_train)
    cm_test = confusion_matrix(y_test, y_pred_test)

    fig, ax = plt.subplots(1, 2, figsize=(11,4))

    print("\nConfusion Matrix:")
    sns.heatmap(cm_train, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], cmap="Blues", fmt='.4g', ax=ax[0])
    ax[0].set_xlabel("Predicted Label")
    ax[0].set_ylabel("True Label")
    ax[0].set_title("Train Confusion Matrix")

    sns.heatmap(cm_test, annot=True, xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'], cmap="Blues", fmt='.4g', ax=ax[1])
    ax[1].set_xlabel("Predicted Label")
    ax[1].set_ylabel("True Label")
    ax[1].set_title("Test Confusion Matrix")

    plt.tight_layout()
    plt.show()


    # calculate classification report
    cr_train = classification_report(y_train, y_pred_train, output_dict=True)
    cr_test = classification_report(y_test, y_pred_test, output_dict=True)
    print("\nTrain Classification Report:")
    crt = pd.DataFrame(cr_train).T
    print(crt.to_markdown())
    
    # sns.heatmap(pd.DataFrame(cr_train).T.iloc[:, :-1], annot=True, cmap="Blues")
    print("\nTest Classification Report:")
    crt2 = pd.DataFrame(cr_test).T
    print(crt2.to_markdown())
    
    # sns.heatmap(pd.DataFrame(cr_test).T.iloc[:, :-1], annot=True, cmap="Blues")
    precision_train = cr_train['weighted avg']['precision']
    precision_test = cr_test['weighted avg']['precision']

    recall_train = cr_train['weighted avg']['recall']
    recall_test = cr_test['weighted avg']['recall']

    acc_train = accuracy_score(y_true = y_train, y_pred = y_pred_train)
    acc_test = accuracy_score(y_true = y_test, y_pred = y_pred_test)

    F1_train = cr_train['weighted avg']['f1-score']
    F1_test = cr_test['weighted avg']['f1-score']

    model_score = [precision_train, precision_test, recall_train, recall_test, acc_train, acc_test, F1_train, F1_test ]
    return model_score
```


```python
# Create a score dataframe
score = pd.DataFrame(index = ['Precision Train', 'Precision Test','Recall Train','Recall Test','Accuracy Train', 'Accuracy Test', 'F1 macro Train', 'F1 macro Test'])

```

# ML Model - 1 : Logistic regression


```python
# ML Model - 1 Implementation
lr_model = LogisticRegression(fit_intercept=True, max_iter=10000)
# Model is trained (fit) and predicted in the evaluate model
```

## 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.


```python
# Visualizing evaluation Metric Score chart
lr_score = evaluate_model(lr_model, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_62_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    0.972222 | 0.945946 |   0.958904 |  37        |
    | 2            |    0.942857 | 0.970588 |   0.956522 |  34        |
    | accuracy     |    0.971429 | 0.971429 |   0.971429 |   0.971429 |
    | macro avg    |    0.971693 | 0.972178 |   0.971809 | 105        |
    | weighted avg |    0.971708 | 0.971429 |   0.97144  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.764706 | 1        |   0.866667 | 13        |
    | 2            |    1        | 0.75     |   0.857143 | 16        |
    | accuracy     |    0.911111 | 0.911111 |   0.911111 |  0.911111 |
    | macro avg    |    0.921569 | 0.916667 |   0.907937 | 45        |
    | weighted avg |    0.932026 | 0.911111 |   0.910688 | 45        |
    


```python
# Updated Evaluation metric Score Chart
score['logistic_regression'] = lr_score
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 1 Implementation with hyperparameter optimization techniques
#(i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
param_grid = {'C': [100,10,1,0.1,0.01,0.001,0.0001],
              'penalty': ['l1', 'l2'],
              'solver':['newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga']}

# Initializing the logistic regression model
logreg = LogisticRegression(fit_intercept=True, max_iter=10000, random_state=0)

# Repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=4, random_state=0)

# Using GridSearchCV to tune the hyperparameters using cross-validation
grid = GridSearchCV(logreg, param_grid, cv=rskf)
grid.fit(x_train, y_train)

# Select the best hyperparameters found by GridSearchCV
best_params = grid.best_params_
print("Best hyperparameters: ", best_params)
```

    Best hyperparameters:  {'C': 10, 'penalty': 'l1', 'solver': 'saga'}
    


```python
# Initiate model with best parameters
lr_model2 = LogisticRegression(C=best_params['C'],
                                  penalty=best_params['penalty'],
                                  solver=best_params['solver'],
                                  max_iter=10000, random_state=0)
```


```python
# Visualizing evaluation Metric Score chart
lr_score2 = evaluate_model(lr_model2, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_67_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    1        | 0.972973 |   0.986301 |  37        |
    | 2            |    0.971429 | 1        |   0.985507 |  34        |
    | accuracy     |    0.990476 | 0.990476 |   0.990476 |   0.990476 |
    | macro avg    |    0.990476 | 0.990991 |   0.990603 | 105        |
    | weighted avg |    0.990748 | 0.990476 |   0.99048  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.928571 | 1        |   0.962963 | 13        |
    | 2            |    1        | 0.9375   |   0.967742 | 16        |
    | accuracy     |    0.977778 | 0.977778 |   0.977778 |  0.977778 |
    | macro avg    |    0.97619  | 0.979167 |   0.976902 | 45        |
    | weighted avg |    0.979365 | 0.977778 |   0.977831 | 45        |
    


```python
score['logistic_regression tuned'] = lr_score2
```


```python
# Updated Evaluation metric Score Chart
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 2 : Decision Tree


```python
# ML Model - 2 Implementation
dt_model = DecisionTreeClassifier(random_state=20)
# Model is trained (fit) and predicted in the evaluate model
```

# 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.


```python
# Visualizing evaluation Metric Score chart
dt_score = evaluate_model(dt_model, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_73_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |           1 |        1 |          1 |        34 |
    | 1            |           1 |        1 |          1 |        37 |
    | 2            |           1 |        1 |          1 |        34 |
    | accuracy     |           1 |        1 |          1 |         1 |
    | macro avg    |           1 |        1 |          1 |       105 |
    | weighted avg |           1 |        1 |          1 |       105 |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.866667 | 1        |   0.928571 | 13        |
    | 2            |    1        | 0.875    |   0.933333 | 16        |
    | accuracy     |    0.955556 | 0.955556 |   0.955556 |  0.955556 |
    | macro avg    |    0.955556 | 0.958333 |   0.953968 | 45        |
    | weighted avg |    0.961481 | 0.955556 |   0.955661 | 45        |
    


```python
# Updated Evaluation metric Score Chart
score['decision_tree'] = dt_score
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 2 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
grid = {'max_depth' : [3,4,5,6,7,8],
        'min_samples_split' : np.arange(2,8),
        'min_samples_leaf' : np.arange(10,20)}

# Initialize the model
model = DecisionTreeClassifier()

# repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

# Initialize GridSearchCV
grid_search = GridSearchCV(model, grid, cv=rskf)

# Fit the GridSearchCV to the training data
grid_search.fit(x_train, y_train)

# Select the best hyperparameters
best_params = grid_search.best_params_
print("best_hyperparameters: ", best_params)
```

    best_hyperparameters:  {'max_depth': 3, 'min_samples_leaf': 10, 'min_samples_split': 3}
    


```python
# Train a new model with the best hyperparameters
dt_model2 = DecisionTreeClassifier(max_depth=best_params['max_depth'],
                                 min_samples_leaf=best_params['min_samples_leaf'],
                                 min_samples_split=best_params['min_samples_split'],
                                 random_state=20)
```


```python
# Visualizing evaluation Metric Score chart
dt2_score = evaluate_model(dt_model2, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_78_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    0.972222 | 0.945946 |   0.958904 |  37        |
    | 2            |    0.942857 | 0.970588 |   0.956522 |  34        |
    | accuracy     |    0.971429 | 0.971429 |   0.971429 |   0.971429 |
    | macro avg    |    0.971693 | 0.972178 |   0.971809 | 105        |
    | weighted avg |    0.971708 | 0.971429 |   0.97144  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.846154 | 0.846154 |   0.846154 | 13        |
    | 2            |    0.875    | 0.875    |   0.875    | 16        |
    | accuracy     |    0.911111 | 0.911111 |   0.911111 |  0.911111 |
    | macro avg    |    0.907051 | 0.907051 |   0.907051 | 45        |
    | weighted avg |    0.911111 | 0.911111 |   0.911111 | 45        |
    


```python
score['decision_tree_tuned'] = dt2_score
```


```python
# Updated Evaluation metric Score Chart
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 3 : Random Forest


```python
# ML Model - 3 Implementation
rf_model = RandomForestClassifier(random_state=0)
# Model is trained (fit) and predicted in the evaluate model
```

# 1. Explain the ML Model used and it's performance using Evaluation Metric Score Chart.


```python
#visualization evaluation metric score chart
rf_score = evaluate_model(rf_model, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_84_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |           1 |        1 |          1 |        34 |
    | 1            |           1 |        1 |          1 |        37 |
    | 2            |           1 |        1 |          1 |        34 |
    | accuracy     |           1 |        1 |          1 |         1 |
    | macro avg    |           1 |        1 |          1 |       105 |
    | weighted avg |           1 |        1 |          1 |       105 |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.764706 | 1        |   0.866667 | 13        |
    | 2            |    1        | 0.75     |   0.857143 | 16        |
    | accuracy     |    0.911111 | 0.911111 |   0.911111 |  0.911111 |
    | macro avg    |    0.921569 | 0.916667 |   0.907937 | 45        |
    | weighted avg |    0.932026 | 0.911111 |   0.910688 | 45        |
    


```python
# Updated Evaluation metric Score Chart
score['random_forest'] = rf_score
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 3 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
grid = {'n_estimators': [10, 50, 100, 200],
              'max_depth': [8, 9, 10, 11, 12,13, 14, 15],
              'min_samples_split': [2, 3, 4, 5]}

# Initialize the model
rf = RandomForestClassifier(random_state=0)

# Repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

# Initialize RandomSearchCV
random_search = RandomizedSearchCV(rf, grid,cv=rskf, n_iter=10, n_jobs=-1)

# Fit the RandomSearchCV to the training data
random_search.fit(x_train, y_train)

# Select the best hyperparameters
best_params = random_search.best_params_
print("best_hyperparameters: ", best_params)
```

    best_hyperparameters:  {'n_estimators': 100, 'min_samples_split': 2, 'max_depth': 9}
    


```python
# Initialize model with best parameters
rf_model2 = RandomForestClassifier(n_estimators = best_params['n_estimators'],
                                 min_samples_leaf= best_params['min_samples_split'],
                                 max_depth = best_params['max_depth'],
                                 random_state=0)
```


```python
# Visualizing evaluation Metric Score chart
rf2_score = evaluate_model(rf_model2, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_89_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    1        | 0.945946 |   0.972222 |  37        |
    | 2            |    0.944444 | 1        |   0.971429 |  34        |
    | accuracy     |    0.980952 | 0.980952 |   0.980952 |   0.980952 |
    | macro avg    |    0.981481 | 0.981982 |   0.981217 | 105        |
    | weighted avg |    0.982011 | 0.980952 |   0.98096  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.8125   | 1        |   0.896552 | 13        |
    | 2            |    1        | 0.8125   |   0.896552 | 16        |
    | accuracy     |    0.933333 | 0.933333 |   0.933333 |  0.933333 |
    | macro avg    |    0.9375   | 0.9375   |   0.931034 | 45        |
    | weighted avg |    0.945833 | 0.933333 |   0.933333 | 45        |
    


```python
score['random_forest_tuned'] = rf2_score
#Updated Evaluation metric Score Chart
score

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
      <td>0.982011</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
      <td>0.945833</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
      <td>0.980960</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
      <td>0.933333</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 4 : SVM (Support Vector Machine)


```python
# ML Model - 4 Implementation
svm_model = SVC(kernel='linear', random_state=0, probability=True)
# Model is trained (fit) and predicted in the evaluate model
```

# 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.


```python
# Visualizing evaluation Metric Score chart
svm_score = evaluate_model(svm_model, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_94_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    0.972222 | 0.945946 |   0.958904 |  37        |
    | 2            |    0.942857 | 0.970588 |   0.956522 |  34        |
    | accuracy     |    0.971429 | 0.971429 |   0.971429 |   0.971429 |
    | macro avg    |    0.971693 | 0.972178 |   0.971809 | 105        |
    | weighted avg |    0.971708 | 0.971429 |   0.97144  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.928571 | 1        |   0.962963 | 13        |
    | 2            |    1        | 0.9375   |   0.967742 | 16        |
    | accuracy     |    0.977778 | 0.977778 |   0.977778 |  0.977778 |
    | macro avg    |    0.97619  | 0.979167 |   0.976902 | 45        |
    | weighted avg |    0.979365 | 0.977778 |   0.977831 | 45        |
    


```python
#Updated Evaluation metric Score Chart
score['s_v_m'] = svm_score
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
      <td>0.982011</td>
      <td>0.971708</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
      <td>0.945833</td>
      <td>0.979365</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
      <td>0.980960</td>
      <td>0.971440</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
      <td>0.933333</td>
      <td>0.977831</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 4 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
param_grid = {'C': np.arange(0.1, 10, 0.1),
              'kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
              'degree': np.arange(2, 6, 1)}

# Initialize the model
svm = SVC(random_state=0, probability=True)

# Repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

# Initialize RandomizedSearchCV with kfold cross-validation
random_search = RandomizedSearchCV(svm, param_grid, n_iter=10, cv=rskf, n_jobs=-1)

# Fit the RandomizedSearchCV to the training data
random_search.fit(x_train, y_train)

# Select the best hyperparameters
best_params = random_search.best_params_
print("best_hyperparameters: ", best_params)
```

    best_hyperparameters:  {'kernel': 'rbf', 'degree': 4, 'C': 6.1}
    


```python
# Initialize model with best parameters
svm_model2 = SVC(C = best_params['C'],
           kernel = best_params['kernel'],
           degree = best_params['degree'],
           random_state=0, probability=True)
```


```python
# Visualizing evaluation Metric Score chart
svm2_score = evaluate_model(svm_model2, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_99_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    1        | 0.945946 |   0.972222 |  37        |
    | 2            |    0.944444 | 1        |   0.971429 |  34        |
    | accuracy     |    0.980952 | 0.980952 |   0.980952 |   0.980952 |
    | macro avg    |    0.981481 | 0.981982 |   0.981217 | 105        |
    | weighted avg |    0.982011 | 0.980952 |   0.98096  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.928571 | 1        |   0.962963 | 13        |
    | 2            |    1        | 0.9375   |   0.967742 | 16        |
    | accuracy     |    0.977778 | 0.977778 |   0.977778 |  0.977778 |
    | macro avg    |    0.97619  | 0.979167 |   0.976902 | 45        |
    | weighted avg |    0.979365 | 0.977778 |   0.977831 | 45        |
    


```python
score['s_v_m_tuned'] = svm2_score
# Updated Evaluation metric Score Chart
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>s_v_m_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
      <td>0.982011</td>
      <td>0.971708</td>
      <td>0.982011</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
      <td>0.945833</td>
      <td>0.979365</td>
      <td>0.979365</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
      <td>0.980960</td>
      <td>0.971440</td>
      <td>0.980960</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
      <td>0.933333</td>
      <td>0.977831</td>
      <td>0.977831</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 5 : Xtreme Gradient Boosting


```python
# ML Model - 5 Implementation
xgb_model = xgb.XGBClassifier()
# Model is trained (fit) and predicted in the evaluate model
```

# 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.


```python
#Visualizing evaluation Metric Score chart
xgb_score = evaluate_model(xgb_model, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_104_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |           1 |        1 |          1 |        34 |
    | 1            |           1 |        1 |          1 |        37 |
    | 2            |           1 |        1 |          1 |        34 |
    | accuracy     |           1 |        1 |          1 |         1 |
    | macro avg    |           1 |        1 |          1 |       105 |
    | weighted avg |           1 |        1 |          1 |       105 |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.8125   | 1        |   0.896552 | 13        |
    | 2            |    1        | 0.8125   |   0.896552 | 16        |
    | accuracy     |    0.933333 | 0.933333 |   0.933333 |  0.933333 |
    | macro avg    |    0.9375   | 0.9375   |   0.931034 | 45        |
    | weighted avg |    0.945833 | 0.933333 |   0.933333 | 45        |
    


```python
# Updated Evaluation metric Score Chart
score['x_g_b'] = xgb_score
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>s_v_m_tuned</th>
      <th>x_g_b</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
      <td>0.982011</td>
      <td>0.971708</td>
      <td>0.982011</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
      <td>0.945833</td>
      <td>0.979365</td>
      <td>0.979365</td>
      <td>0.945833</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
      <td>0.980960</td>
      <td>0.971440</td>
      <td>0.980960</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
      <td>0.933333</td>
      <td>0.977831</td>
      <td>0.977831</td>
      <td>0.933333</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 5 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
param_grid = {'learning_rate': np.arange(0.01, 0.3, 0.01),
              'max_depth': np.arange(3, 15, 1),
              'n_estimators': np.arange(100, 200, 10)}

# Initialize the model
xgb2 = xgb.XGBClassifier(random_state=0)

# Repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(xgb2, param_grid, n_iter=10, cv=rskf)

# Fit the RandomizedSearchCV to the training data
random_search.fit(x_train, y_train)

# Select the best hyperparameters
best_params = random_search.best_params_
print("best_hyperparameters: ", best_params)
```

    best_hyperparameters:  {'n_estimators': 130, 'max_depth': 4, 'learning_rate': 0.01}
    


```python
# Initialize model with best parameters
xgb_model2 = xgb.XGBClassifier(learning_rate = best_params['learning_rate'],
                                 max_depth = best_params['max_depth'],
                               n_estimators = best_params['n_estimators'],
                                 random_state=0)
```


```python
# Visualizing evaluation Metric Score chart
xgb2_score = evaluate_model(xgb_model2, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_109_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    1        | 0.972973 |   0.986301 |  37        |
    | 2            |    0.971429 | 1        |   0.985507 |  34        |
    | accuracy     |    0.990476 | 0.990476 |   0.990476 |   0.990476 |
    | macro avg    |    0.990476 | 0.990991 |   0.990603 | 105        |
    | weighted avg |    0.990748 | 0.990476 |   0.99048  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.866667 | 1        |   0.928571 | 13        |
    | 2            |    1        | 0.875    |   0.933333 | 16        |
    | accuracy     |    0.955556 | 0.955556 |   0.955556 |  0.955556 |
    | macro avg    |    0.955556 | 0.958333 |   0.953968 | 45        |
    | weighted avg |    0.961481 | 0.955556 |   0.955661 | 45        |
    


```python
score['x_g_b_tuned'] = xgb2_score
# Updated Evaluation metric Score Chart
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>s_v_m_tuned</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
      <td>0.982011</td>
      <td>0.971708</td>
      <td>0.982011</td>
      <td>1.000000</td>
      <td>0.990748</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
      <td>0.945833</td>
      <td>0.979365</td>
      <td>0.979365</td>
      <td>0.945833</td>
      <td>0.961481</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
      <td>0.980960</td>
      <td>0.971440</td>
      <td>0.980960</td>
      <td>1.000000</td>
      <td>0.990480</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
      <td>0.933333</td>
      <td>0.977831</td>
      <td>0.977831</td>
      <td>0.933333</td>
      <td>0.955661</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 6 : Naive Bayes


```python
# ML Model - 6 Implementation
nb_model = GaussianNB()

# Model is trained (fit) and predicted in the evaluate model
```


```python
# Visualizing evaluation Metric Score chart
nb_score = evaluate_model(nb_model, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_113_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    0.972222 | 0.945946 |   0.958904 |  37        |
    | 2            |    0.942857 | 0.970588 |   0.956522 |  34        |
    | accuracy     |    0.971429 | 0.971429 |   0.971429 |   0.971429 |
    | macro avg    |    0.971693 | 0.972178 |   0.971809 | 105        |
    | weighted avg |    0.971708 | 0.971429 |   0.97144  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.8      | 0.923077 |   0.857143 | 13        |
    | 2            |    0.928571 | 0.8125   |   0.866667 | 16        |
    | accuracy     |    0.911111 | 0.911111 |   0.911111 |  0.911111 |
    | macro avg    |    0.909524 | 0.911859 |   0.907937 | 45        |
    | weighted avg |    0.916825 | 0.911111 |   0.911323 | 45        |
    


```python
# Updated Evaluation metric Score Chart
score['naive_bayes'] = nb_score
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>s_v_m_tuned</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
      <th>naive_bayes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
      <td>0.982011</td>
      <td>0.971708</td>
      <td>0.982011</td>
      <td>1.000000</td>
      <td>0.990748</td>
      <td>0.971708</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
      <td>0.945833</td>
      <td>0.979365</td>
      <td>0.979365</td>
      <td>0.945833</td>
      <td>0.961481</td>
      <td>0.916825</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
      <td>0.980960</td>
      <td>0.971440</td>
      <td>0.980960</td>
      <td>1.000000</td>
      <td>0.990480</td>
      <td>0.971440</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
      <td>0.933333</td>
      <td>0.977831</td>
      <td>0.977831</td>
      <td>0.933333</td>
      <td>0.955661</td>
      <td>0.911323</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 6 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
param_grid = {'var_smoothing': np.logspace(0,-9, num=100)}

# Initialize the model
naive = GaussianNB()

# repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=4, n_repeats=4, random_state=0)

# Initialize GridSearchCV
GridSearch = GridSearchCV(naive, param_grid, cv=rskf, n_jobs=-1)

# Fit the GridSearchCV to the training data
GridSearch.fit(x_train, y_train)

# Select the best hyperparameters
best_params = GridSearch.best_params_
print("best_hyperparameters: ", best_params)
```

    best_hyperparameters:  {'var_smoothing': 0.01519911082952933}
    


```python
# Initiate model with best parameters
nb_model2 = GaussianNB(var_smoothing = best_params['var_smoothing'])
```


```python
# Visualizing evaluation Metric Score chart
nb2_score = evaluate_model(nb_model2, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_118_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    1        | 0.945946 |   0.972222 |  37        |
    | 2            |    0.944444 | 1        |   0.971429 |  34        |
    | accuracy     |    0.980952 | 0.980952 |   0.980952 |   0.980952 |
    | macro avg    |    0.981481 | 0.981982 |   0.981217 | 105        |
    | weighted avg |    0.982011 | 0.980952 |   0.98096  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.75     | 0.923077 |   0.827586 | 13        |
    | 2            |    0.923077 | 0.75     |   0.827586 | 16        |
    | accuracy     |    0.888889 | 0.888889 |   0.888889 |  0.888889 |
    | macro avg    |    0.891026 | 0.891026 |   0.885057 | 45        |
    | weighted avg |    0.900427 | 0.888889 |   0.888889 | 45        |
    


```python
score['naive_bayes_tuned']= nb2_score
# Updated Evaluation metric Score Chart
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>s_v_m_tuned</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
      <th>naive_bayes</th>
      <th>naive_bayes_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
      <td>0.982011</td>
      <td>0.971708</td>
      <td>0.982011</td>
      <td>1.000000</td>
      <td>0.990748</td>
      <td>0.971708</td>
      <td>0.982011</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
      <td>0.945833</td>
      <td>0.979365</td>
      <td>0.979365</td>
      <td>0.945833</td>
      <td>0.961481</td>
      <td>0.916825</td>
      <td>0.900427</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
      <td>0.971429</td>
      <td>0.980952</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
      <td>0.971429</td>
      <td>0.980952</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.888889</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
      <td>0.980960</td>
      <td>0.971440</td>
      <td>0.980960</td>
      <td>1.000000</td>
      <td>0.990480</td>
      <td>0.971440</td>
      <td>0.980960</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
      <td>0.933333</td>
      <td>0.977831</td>
      <td>0.977831</td>
      <td>0.933333</td>
      <td>0.955661</td>
      <td>0.911323</td>
      <td>0.888889</td>
    </tr>
  </tbody>
</table>
</div>



# ML Model - 7 : Neural Network


```python
# ML Model - 7 Implementation
nn_model = MLPClassifier(random_state=0)
# Model is trained (fit) and predicted in the evaluate model
```

# 1. Explain the ML Model used and it's performance using Evaluation metric Score Chart.


```python
# Visualizing evaluation Metric Score chart
neural_score = evaluate_model(nn_model, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_123_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    1        | 0.918919 |   0.957746 |  37        |
    | 2            |    0.918919 | 1        |   0.957746 |  34        |
    | accuracy     |    0.971429 | 0.971429 |   0.971429 |   0.971429 |
    | macro avg    |    0.972973 | 0.972973 |   0.971831 | 105        |
    | weighted avg |    0.973745 | 0.971429 |   0.971429 | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    0.923077 | 0.923077 |   0.923077 | 13        |
    | 2            |    0.9375   | 0.9375   |   0.9375   | 16        |
    | accuracy     |    0.955556 | 0.955556 |   0.955556 |  0.955556 |
    | macro avg    |    0.953526 | 0.953526 |   0.953526 | 45        |
    | weighted avg |    0.955556 | 0.955556 |   0.955556 | 45        |
    


```python
# Updated Evaluation metric Score Chart
score['neural_network'] = neural_score
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>s_v_m_tuned</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
      <th>naive_bayes</th>
      <th>naive_bayes_tuned</th>
      <th>neural_network</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
      <td>0.982011</td>
      <td>0.971708</td>
      <td>0.982011</td>
      <td>1.000000</td>
      <td>0.990748</td>
      <td>0.971708</td>
      <td>0.982011</td>
      <td>0.973745</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
      <td>0.945833</td>
      <td>0.979365</td>
      <td>0.979365</td>
      <td>0.945833</td>
      <td>0.961481</td>
      <td>0.916825</td>
      <td>0.900427</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.888889</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.888889</td>
      <td>0.955556</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
      <td>0.980960</td>
      <td>0.971440</td>
      <td>0.980960</td>
      <td>1.000000</td>
      <td>0.990480</td>
      <td>0.971440</td>
      <td>0.980960</td>
      <td>0.971429</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
      <td>0.933333</td>
      <td>0.977831</td>
      <td>0.977831</td>
      <td>0.933333</td>
      <td>0.955661</td>
      <td>0.911323</td>
      <td>0.888889</td>
      <td>0.955556</td>
    </tr>
  </tbody>
</table>
</div>



# 2. Cross- Validation & Hyperparameter Tuning


```python
# ML Model - 7 Implementation with hyperparameter optimization techniques (i.e., GridSearch CV, RandomSearch CV, Bayesian Optimization etc.)
# Define the hyperparameter grid
param_grid = {'hidden_layer_sizes': np.arange(10, 100, 10),
              'alpha': np.arange(0.0001, 0.01, 0.0001)}

# Initialize the model
neural = MLPClassifier(random_state=0)

# Repeated stratified kfold
rskf = RepeatedStratifiedKFold(n_splits=3, n_repeats=3, random_state=0)

# Initialize RandomizedSearchCV
random_search = RandomizedSearchCV(neural, param_grid, n_iter=10, cv=rskf, n_jobs=-1)

# Fit the RandomizedSearchCV to the training data
random_search.fit(x_train, y_train)

# Select the best hyperparameters
best_params = random_search.best_params_
print("best_hyperparameters: ", best_params)
```

    best_hyperparameters:  {'hidden_layer_sizes': 40, 'alpha': 0.008}
    


```python
# Initiate model with best hyperparameters
nn_model2 = MLPClassifier(hidden_layer_sizes = best_params['hidden_layer_sizes'],
                        alpha = best_params['alpha'],
                        random_state = 0)
```


```python
# Visualizing evaluation Metric Score chart
neural2_score = evaluate_model(nn_model2, x_train, x_test, y_train, y_test)
```

    
    Confusion Matrix:
    


    
![png](output_128_1.png)
    


    
    Train Classification Report:
    |              |   precision |   recall |   f1-score |    support |
    |:-------------|------------:|---------:|-----------:|-----------:|
    | 0            |    1        | 1        |   1        |  34        |
    | 1            |    1        | 0.945946 |   0.972222 |  37        |
    | 2            |    0.944444 | 1        |   0.971429 |  34        |
    | accuracy     |    0.980952 | 0.980952 |   0.980952 |   0.980952 |
    | macro avg    |    0.981481 | 0.981982 |   0.981217 | 105        |
    | weighted avg |    0.982011 | 0.980952 |   0.98096  | 105        |
    
    Test Classification Report:
    |              |   precision |   recall |   f1-score |   support |
    |:-------------|------------:|---------:|-----------:|----------:|
    | 0            |    1        | 1        |   1        | 16        |
    | 1            |    1        | 0.923077 |   0.96     | 13        |
    | 2            |    0.941176 | 1        |   0.969697 | 16        |
    | accuracy     |    0.977778 | 0.977778 |   0.977778 |  0.977778 |
    | macro avg    |    0.980392 | 0.974359 |   0.976566 | 45        |
    | weighted avg |    0.979085 | 0.977778 |   0.97767  | 45        |
    


```python
score['neural_network_tuned']= neural2_score
# Updated Evaluation metric Score Chart
score
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>logistic_regression</th>
      <th>logistic_regression tuned</th>
      <th>decision_tree</th>
      <th>decision_tree_tuned</th>
      <th>random_forest</th>
      <th>random_forest_tuned</th>
      <th>s_v_m</th>
      <th>s_v_m_tuned</th>
      <th>x_g_b</th>
      <th>x_g_b_tuned</th>
      <th>naive_bayes</th>
      <th>naive_bayes_tuned</th>
      <th>neural_network</th>
      <th>neural_network_tuned</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Precision Train</th>
      <td>0.971708</td>
      <td>0.990748</td>
      <td>1.000000</td>
      <td>0.971708</td>
      <td>1.000000</td>
      <td>0.982011</td>
      <td>0.971708</td>
      <td>0.982011</td>
      <td>1.000000</td>
      <td>0.990748</td>
      <td>0.971708</td>
      <td>0.982011</td>
      <td>0.973745</td>
      <td>0.982011</td>
    </tr>
    <tr>
      <th>Precision Test</th>
      <td>0.932026</td>
      <td>0.979365</td>
      <td>0.961481</td>
      <td>0.911111</td>
      <td>0.932026</td>
      <td>0.945833</td>
      <td>0.979365</td>
      <td>0.979365</td>
      <td>0.945833</td>
      <td>0.961481</td>
      <td>0.916825</td>
      <td>0.900427</td>
      <td>0.955556</td>
      <td>0.979085</td>
    </tr>
    <tr>
      <th>Recall Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
    </tr>
    <tr>
      <th>Recall Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.888889</td>
      <td>0.955556</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>Accuracy Train</th>
      <td>0.971429</td>
      <td>0.990476</td>
      <td>1.000000</td>
      <td>0.971429</td>
      <td>1.000000</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>1.000000</td>
      <td>0.990476</td>
      <td>0.971429</td>
      <td>0.980952</td>
      <td>0.971429</td>
      <td>0.980952</td>
    </tr>
    <tr>
      <th>Accuracy Test</th>
      <td>0.911111</td>
      <td>0.977778</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.911111</td>
      <td>0.933333</td>
      <td>0.977778</td>
      <td>0.977778</td>
      <td>0.933333</td>
      <td>0.955556</td>
      <td>0.911111</td>
      <td>0.888889</td>
      <td>0.955556</td>
      <td>0.977778</td>
    </tr>
    <tr>
      <th>F1 macro Train</th>
      <td>0.971440</td>
      <td>0.990480</td>
      <td>1.000000</td>
      <td>0.971440</td>
      <td>1.000000</td>
      <td>0.980960</td>
      <td>0.971440</td>
      <td>0.980960</td>
      <td>1.000000</td>
      <td>0.990480</td>
      <td>0.971440</td>
      <td>0.980960</td>
      <td>0.971429</td>
      <td>0.980960</td>
    </tr>
    <tr>
      <th>F1 macro Test</th>
      <td>0.910688</td>
      <td>0.977831</td>
      <td>0.955661</td>
      <td>0.911111</td>
      <td>0.910688</td>
      <td>0.933333</td>
      <td>0.977831</td>
      <td>0.977831</td>
      <td>0.933333</td>
      <td>0.955661</td>
      <td>0.911323</td>
      <td>0.888889</td>
      <td>0.955556</td>
      <td>0.977670</td>
    </tr>
  </tbody>
</table>
</div>



# Markdown


```python
print(score.to_markdown())
```

    |                 |   logistic_regression |   logistic_regression tuned |   decision_tree |   decision_tree_tuned |   random_forest |   random_forest_tuned |    s_v_m |   s_v_m_tuned |    x_g_b |   x_g_b_tuned |   naive_bayes |   naive_bayes_tuned |   neural_network |   neural_network_tuned |
    |:----------------|----------------------:|----------------------------:|----------------:|----------------------:|----------------:|----------------------:|---------:|--------------:|---------:|--------------:|--------------:|--------------------:|-----------------:|-----------------------:|
    | Precision Train |              0.971708 |                    0.990748 |        1        |              0.971708 |        1        |              0.982011 | 0.971708 |      0.982011 | 1        |      0.990748 |      0.971708 |            0.982011 |         0.973745 |               0.982011 |
    | Precision Test  |              0.932026 |                    0.979365 |        0.961481 |              0.911111 |        0.932026 |              0.945833 | 0.979365 |      0.979365 | 0.945833 |      0.961481 |      0.916825 |            0.900427 |         0.955556 |               0.979085 |
    | Recall Train    |              0.971429 |                    0.990476 |        1        |              0.971429 |        1        |              0.980952 | 0.971429 |      0.980952 | 1        |      0.990476 |      0.971429 |            0.980952 |         0.971429 |               0.980952 |
    | Recall Test     |              0.911111 |                    0.977778 |        0.955556 |              0.911111 |        0.911111 |              0.933333 | 0.977778 |      0.977778 | 0.933333 |      0.955556 |      0.911111 |            0.888889 |         0.955556 |               0.977778 |
    | Accuracy Train  |              0.971429 |                    0.990476 |        1        |              0.971429 |        1        |              0.980952 | 0.971429 |      0.980952 | 1        |      0.990476 |      0.971429 |            0.980952 |         0.971429 |               0.980952 |
    | Accuracy Test   |              0.911111 |                    0.977778 |        0.955556 |              0.911111 |        0.911111 |              0.933333 | 0.977778 |      0.977778 | 0.933333 |      0.955556 |      0.911111 |            0.888889 |         0.955556 |               0.977778 |
    | F1 macro Train  |              0.97144  |                    0.99048  |        1        |              0.97144  |        1        |              0.98096  | 0.97144  |      0.98096  | 1        |      0.99048  |      0.97144  |            0.98096  |         0.971429 |               0.98096  |
    | F1 macro Test   |              0.910688 |                    0.977831 |        0.955661 |              0.911111 |        0.910688 |              0.933333 | 0.977831 |      0.977831 | 0.933333 |      0.955661 |      0.911323 |            0.888889 |         0.955556 |               0.97767  |
    

# *Selection of best model*


```python
# Removing the overfitted models which have precision, recall, f1 scores for train as 1
score_t = score.transpose()            # taking transpose of the score dataframe to create new difference column
remove_models = score_t[score_t['Recall Train']>=0.98].index  # creating a list of models which have 1 for train and score_t['Accuracy Train']==1.0 and score_t['Precision Train']==1.0 and score_t['F1 macro Train']==1.0
remove_models

abc = score_t.drop(remove_models)                     # creating a new dataframe with required models
abc
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Precision Train</th>
      <th>Precision Test</th>
      <th>Recall Train</th>
      <th>Recall Test</th>
      <th>Accuracy Train</th>
      <th>Accuracy Test</th>
      <th>F1 macro Train</th>
      <th>F1 macro Test</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>logistic_regression</th>
      <td>0.971708</td>
      <td>0.932026</td>
      <td>0.971429</td>
      <td>0.911111</td>
      <td>0.971429</td>
      <td>0.911111</td>
      <td>0.971440</td>
      <td>0.910688</td>
    </tr>
    <tr>
      <th>decision_tree_tuned</th>
      <td>0.971708</td>
      <td>0.911111</td>
      <td>0.971429</td>
      <td>0.911111</td>
      <td>0.971429</td>
      <td>0.911111</td>
      <td>0.971440</td>
      <td>0.911111</td>
    </tr>
    <tr>
      <th>s_v_m</th>
      <td>0.971708</td>
      <td>0.979365</td>
      <td>0.971429</td>
      <td>0.977778</td>
      <td>0.971429</td>
      <td>0.977778</td>
      <td>0.971440</td>
      <td>0.977831</td>
    </tr>
    <tr>
      <th>naive_bayes</th>
      <td>0.971708</td>
      <td>0.916825</td>
      <td>0.971429</td>
      <td>0.911111</td>
      <td>0.971429</td>
      <td>0.911111</td>
      <td>0.971440</td>
      <td>0.911323</td>
    </tr>
    <tr>
      <th>neural_network</th>
      <td>0.973745</td>
      <td>0.955556</td>
      <td>0.971429</td>
      <td>0.955556</td>
      <td>0.971429</td>
      <td>0.955556</td>
      <td>0.971429</td>
      <td>0.955556</td>
    </tr>
  </tbody>
</table>
</div>




```python
def select_best_model(df, metrics):

    best_models = {}
    for metric in metrics:
        max_test = df[metric + ' Test'].max()
        best_model_test = df[df[metric + ' Test'] == max_test].index[0]
        best_model = best_model_test
        best_models[metric] = best_model
    return best_models
```


```python
metrics = ['Precision', 'Recall', 'Accuracy', 'F1 macro']
best_models = select_best_model(abc, metrics)
print("The best models are:")
for metric, best_model in best_models.items():
    print(f"{metric}: {best_model} - {abc[metric+' Test'][best_model].round(4)}")
```

    The best models are:
    Precision: s_v_m - 0.9794
    Recall: s_v_m - 0.9778
    Accuracy: s_v_m - 0.9778
    F1 macro: s_v_m - 0.9778
    


```python
# Take recall as the primary evaluation metric
score_smpl = score.transpose()
remove_overfitting_models = score_smpl[score_smpl['Recall Train']>=0.98].index
remove_overfitting_models
new_score = score_smpl.drop(remove_overfitting_models)
new_score = new_score.drop(['Precision Train','Precision Test','Accuracy Train','Accuracy Test','F1 macro Train','F1 macro Test'], axis=1)
new_score.index.name = 'Classification Model'
print(new_score.to_markdown())
```

    | Classification Model   |   Recall Train |   Recall Test |
    |:-----------------------|---------------:|--------------:|
    | logistic_regression    |       0.971429 |      0.911111 |
    | decision_tree_tuned    |       0.971429 |      0.911111 |
    | s_v_m                  |       0.971429 |      0.977778 |
    | naive_bayes            |       0.971429 |      0.911111 |
    | neural_network         |       0.971429 |      0.955556 |
    

# Explain the model which have used for the prediction


```python
# Define a list of category labels for reference.
Category_RF = ['Iris-Setosa', 'Iris-Versicolor', 'Iris-Virginica']
```


```python
# In this example, it's a data point with Sepal Length, Sepal Width, Petal Length, and Petal Width.
x_rf = np.array([[5.1, 3.5, 1.4, 0.2]])

# Use the tuned random forest model (rf_model2) to make a prediction.
x_rf_prediction = rf_model2.predict(x_rf)
x_rf_prediction[0]

# Display the predicted category label.
print(Category_RF[int(x_rf_prediction[0])])
```

    Iris-Setosa
    

# Testing Accuracy


```python
# Evaluate the model 
accuracy = rf_model.score(x_test, y_test)
print(f"Model Accuracy: {accuracy}")

```

    Model Accuracy: 0.9111111111111111
    

# Saving iris_flower_model.pkl as file


```python
# After training, add this code to save the model:
import pickle
with open('iris_flower_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)  # Save the trained model to a file

# Once you run the cell with this code, 'iris_flower_model.pkl' will be created in your folder

```
