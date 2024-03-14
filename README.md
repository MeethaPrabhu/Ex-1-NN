<H3>Meetha Prabhu</H3>
<H3>212222240065</H3>
<H3>EX. NO.1</H3>
<H3>20.02.2024</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```python
from google.colab import files
import pandas as pd
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

df=pd.read_csv("/content/Churn_Modelling.csv")

print(df)

x=df.iloc[:,:-1].values

print(x)

y=df.iloc[:,-1].values

print(y)

print(df.isnull().sum())

df.fillna(df.mean().round(1), inplace=True)

print(df.isnull().sum())

y=df.iloc[:,-1].values

print(y)

df.duplicated()

print(df['RowNumber'].describe())
print(df['CustomerId'].describe())
print(df['Surname'].describe())
print(df['CreditScore'].describe())
print(df['Geography'].describe())
print(df['Gender'].describe())
print(df['Age'].describe())
print(df['Tenure'].describe())
print(df['Balance'].describe())
print(df['NumOfProducts'].describe())
print(df['HasCrCard'].describe())
print(df['IsActiveMember'].describe())
print(df['EstimatedSalary'].describe())
print(df['Exited'].describe())

df1=df.drop(['Surname','Geography','Gender'],axis=1)

df1.head()
scaler=MinMaxScaler()

df2=pd.DataFrame(scaler.fit_transform(df1))
print(df2)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
print(x_train)

print("Length of x_train:",len(x_train))
print(x_test)

print("Length of x_test: ",len(x_test))
```

## OUTPUT:
Data Frame:
![alt text](image-1.png)

X Values:<br>
![alt text](image-2.png)

Y Values:<br>
![alt text](image-3.png)

Sum of null values:<br>
![alt text](image-4.png)

Duplicated Values:<br>
![alt text](image-5.png)

Row Number:<br>
![alt text](image-6.png)

Customer Id:<br>
![alt text](image-7.png)

Surname:<br>
![alt text](image-8.png)

Credit Score:<br>
![alt text](image-9.png)

Gender:<br>
![alt text](image-10.png)

Age:<br>
![alt text](image-11.png)

Tenure:<br>
![alt text](image-12.png)

Balance:<br>
![alt text](image-13.png)

Number of Products:<br>
![alt text](image-14.png)

Has Credit card:<br>
![alt text](image-15.png)

Is active number:<br>
![alt text](image-16.png)

Estimated Salary:<br>
![alt text](image-17.png)

Exited:<br>
![alt text](image-18.png)

After dropping Surname, Geography and Gender:<br>
![alt text](image-19.png)

Transformed data frame:<br>
![alt text](image-20.png)

Trained x data:<br>
![alt text](image-21.png)

Length of X train data set:<br>
![alt text](image-22.png)

Test data:<br>
![alt text](image-23.png)

Length of test data:<br>
![alt text](image-24.png)
## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


