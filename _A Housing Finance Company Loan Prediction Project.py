#!/usr/bin/env python
# coding: utf-8

# # Loan Prediction Analysis

# Introduction Loan Prediction Problem 
# 
# Below is a brief introduction to this topic
# 
# The Objective of the project is designed for people who want to solve binary classification problems using Python. By the end of this jupyter notebook, you will gain the necessary skills and techniques required to solve such problems.
# 
# Introduction to the problem
# 
# 1. Exploratory Data Analysis (EDA) and Pre-Processing
# 
# 2. Model building and Feature engineering
# 
# Problem Statement :
# 
# Understanding the problem statement is the first and foremost step. Let us see the problem statement.
# 
# A Housing Finance company deals in all home loans. They have a presence across all urban, semi-urban and rural areas. Customers first apply for a home loan after that company validates the customer’s eligibility for a loan. The company wants to automate the loan eligibility process (real-time) based on customer detail provided while filling out the online application form.
# 
# These details are Gender, Marital Status, Education, Number of Dependents, Income, Loan Amount, Credit History, etc. 
# 
# Problem Statement : It is a classification problem where we have to predict whether a loan would be approved or not. In these kind of problems, we have to predict discrete values based on a given set of independent variables (s).
# 
# Classification can be of two types:
# 
# Binary Classification:- In this, we have to predict either of the two given classes. For example: classifying the “gender” as male or female, predicting the “result” as to win or loss, etc.
# 
# MultiClass Classification:- Here we have to classify the data into three or more classes. For example: classifying a “movie’s genre” as comedy, action, or romantic, classifying “fruits” like oranges, apples, pears, etc.
# 
# Loan prediction is a very common real-life problem that each retail bank faces at least once in its lifetime.
# 
# Hypothesis Generation : After looking at the problem statement, we will now move into hypothesis generation. It is the process of listing out all the possible factors that can affect the outcome.
# 
# What is Hypothesis Generation?
# 
# This is a very important stage in a data science/machine learning pipeline. It involves understanding the problem in detail by brainstorming maximum possibilities that can impact the outcome. It is done by thoroughly understanding the problem statement before looking at the data.
# 
# Below are some of the factors which I think can affect the Loan Approval:
# 
# 1. Salary: Applicants with high income should have more chances of getting approval. 
# 2. Previous history: Applicants who have paid their historical debts have more chances of getting approval. 3. 3. Loan amount: Less the amount higher the chances of getting approval. 
# 4. Loan term: Less the time period has higher chances of approval. 
# 5. EMI: Lesser the amount to be paid monthly, the higher the chances of getting approval. 
# 
# These are some of the factors which I think can affect the target variable, you can find many more factors.
# 
# The Dataset : loan_data.csv

# # Importing Libraries

# In[1]:


import pandas as pd                       # for reading the files
import numpy as np                        # for creating multi-dimensional-array
import matplotlib.pyplot as plt           # for plotting
import seaborn as sns                     # for data visulization
import warnings                           # for ignoring the warnings
warnings.filterwarnings("ignore")
get_ipython().run_line_magic('matplotlib', 'inline')


# # Import the Data Files

# In[2]:


test= pd.read_csv('/Users/ruchishukla/Downloads/test.csv')
train= pd.read_csv('/Users/ruchishukla/Downloads/train.csv')


# Test File

# In[3]:


test.head()


# In[4]:


test.shape


# Training File

# In[5]:


train.head()


# In[6]:


train.shape


# Creating a copy of file so that any changes made doesn't affect the original datasets

# In[7]:


test_original= test.copy()
train_original= train.copy()


# Checking the Data Types of Variables

# In[8]:


test.dtypes


# In[9]:


train.dtypes


# In[ ]:





# # Univariant Analysis
# (Examing each variable individually)

# 1. Target Variable i.e. 'Loan Status'

# In[10]:


train['Loan_Status'].value_counts()                    #counting the values of different Loan Status


# In[11]:


train['Loan_Status'].value_counts().plot.bar()         


# In[12]:


train['Loan_Status'].value_counts(normalize=True).plot.bar()
# normalize = True will give the probability in y-axis

plt.title("Loan Status")


# In[ ]:





# Plots for Independent Categorical Variables

# In[13]:


plt.figure()
plt.subplot(321)
train['Gender'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Gender')

plt.subplot(322)
train['Married'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Married')

plt.subplot(323)
train['Education'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Education')

plt.subplot(324)
train['Self_Employed'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Self-Employed')

plt.subplot(325)
train['Credit_History'].value_counts(normalize=True).plot.bar(figsize=(20,20),title='Credit_History')


# In[ ]:





# Plots for Independent Ordinal Variables

# In[14]:


plt.figure()
plt.subplot(121)
train['Dependents'].value_counts(normalize=True).plot.bar(figsize=(20,5),title='Dependents')

plt.subplot(122)
train['Property_Area'].value_counts(normalize=True).plot.bar(figsize=(20,5),title='Property Area')


# In[ ]:





# Plots for Independent Numerical Variables

# Applicant Income

# In[15]:


plt.subplot(121)
sns.distplot(train['ApplicantIncome'])
plt.subplot(122)
train['ApplicantIncome'].plot.box(figsize=(20,5))


# In[ ]:





# In[16]:


train.boxplot(column='ApplicantIncome',by='Education')
plt.suptitle("")


# In[ ]:





# Co-applicant Income

# In[17]:


plt.subplot(121)
sns.distplot(train['CoapplicantIncome'])
plt.subplot(122)
train['CoapplicantIncome'].plot.box(figsize=(20,5))


# In[ ]:





# In[18]:


df=train.dropna()
plt.subplot(121)
sns.distplot(df['LoanAmount'])

plt.subplot(122)
df['LoanAmount'].plot.box(figsize=(20,5))


# In[ ]:





# # Bivariant Analysis
# (Examing two variables at a time)

# Frequency Table for Gender and Loan Status

# In[19]:


Gender=pd.crosstab(train['Gender'],train['Loan_Status']) 
Gender


# In[20]:


Gender.div(Gender.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[ ]:





# Frequency Table for Married and Loan Status

# In[21]:


Married=pd.crosstab(train['Married'],train['Loan_Status']) 
Married


# In[22]:


Married.div(Married.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[ ]:





# Frequency Table for Dependents and Loan Status

# In[23]:


Dependents=pd.crosstab(train['Dependents'],train['Loan_Status']) 
Dependents


# In[24]:


Dependents.div(Dependents.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True, figsize=(4,4))


# In[ ]:





# Frequency Table for Education and Loan Status

# In[25]:


Education= pd.crosstab(train['Education'],train['Loan_Status'])
Education


# In[26]:


Education.div(Education.sum(1).astype(float),axis=0).plot(kind="bar", stacked=True, figsize=(4,4) )


# In[ ]:





# Frequency Table for Self Employed and Loan Status

# In[27]:


Self_Employed= pd.crosstab(train['Self_Employed'],train['Loan_Status'])
Self_Employed


# In[28]:


Self_Employed.div(Self_Employed.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True,figsize=(4,4))


# In[ ]:





# Frequency Table for Credit History and Loan Status

# In[29]:


Credit_History= pd.crosstab(train['Credit_History'],train['Loan_Status'])
Credit_History


# In[30]:


Credit_History.div(Credit_History.sum(1).astype(float),axis=0).plot(kind="bar",stacked=True, figsize=(4,4))


# In[ ]:





# Frequency Table for Property Area and Loan Status

# In[31]:


Property_Area=pd.crosstab(train['Property_Area'],train['Loan_Status'])
Property_Area


# In[32]:


Property_Area.div(Property_Area.sum(1).astype(float),axis=0).plot(kind='bar', stacked=True, figsize=(4,4))


# In[ ]:





# Plotting of Numerical Categorical Variable and Loan Status

# In[34]:


bins=[0,2500,4000,6000,8100] 
group=['Low','Average','High', 'Very high'] 
train['Income_bin']=pd.cut(train['ApplicantIncome'],bins,labels=group)


# In[35]:


Income_bin=pd.crosstab(train['Income_bin'],train['Loan_Status']) 
Income_bin.div(Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('ApplicantIncome') 
P=plt.ylabel('Percentage')


# In[ ]:





# Doing the same for Coapplicant Income

# In[36]:


bins=[0,1000,3000,42000] 
group=['Low','Average','High'] 
train['Coapplicant_Income_bin']=pd.cut(train['CoapplicantIncome'],bins,labels=group)
Coapplicant_Income_bin=pd.crosstab(train['Coapplicant_Income_bin'],train['Loan_Status']) 
Coapplicant_Income_bin.div(Coapplicant_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('CoapplicantIncome') 
P = plt.ylabel('Percentage')


# It shows that if coapplicant’s income is less the chances of loan approval are high. But this does not look right. The possible reason behind this may be that most of the applicants don’t have any coapplicant so the coapplicant income for such applicants is 0 and hence the loan approval is not dependent on it. So we can make a new variable in which we will combine the applicant’s and coapplicant’s income to visualize the combined effect of income on loan approval.
# 
# Let us combine the Applicant Income and Coapplicant Income and see the combined effect of Total Income on the Loan_Status.

# In[37]:


train['Total_Income']=train['ApplicantIncome']+train['CoapplicantIncome']
bins=[0,2500,4000,6000,81000] 
group=['Low','Average','High', 'Very high'] 
train['Total_Income_bin']=pd.cut(train['Total_Income'],bins,labels=group)
Total_Income_bin=pd.crosstab(train['Total_Income_bin'],train['Loan_Status']) 
Total_Income_bin.div(Total_Income_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('Total_Income') 
P = plt.ylabel('Percentage')


# In[ ]:





# Plotting of Loan Amount and Loan Status

# In[38]:


bins=[0,100,200,700] 
group=['Low','Average','High'] 
train['LoanAmount_bin']=pd.cut(train['LoanAmount'],bins,labels=group)
LoanAmount_bin=pd.crosstab(train['LoanAmount_bin'],train['Loan_Status']) 
LoanAmount_bin.div(LoanAmount_bin.sum(1).astype(float), axis=0).plot(kind="bar", stacked=True) 
plt.xlabel('LoanAmount') 
P = plt.ylabel('Percentage')


# In[ ]:





# Change the 3+ in dependents variable to 3 to make it a numerical variable.We will also convert the target variable’s categories into 0 and 1 

# In[39]:


train['Dependents'].replace('3+', 3,inplace=True) 
test['Dependents'].replace('3+', 3,inplace=True) 


# In[ ]:





# Convert the target variable 'Loan Status' categories into 0 and 1 for logistic regression

# In[40]:


train['Loan_Status'].replace('N', 0,inplace=True) 
train['Loan_Status'].replace('Y', 1,inplace=True)


# In[ ]:





# # Correlation using Heatmaps

# In[41]:


matrix = train.corr() 
plt.figure(figsize=(9,6))
sns.heatmap(matrix, square=True, cmap="BuPu")


# In[42]:


train=train.drop(['Income_bin', 'Coapplicant_Income_bin', 'LoanAmount_bin', 'Total_Income_bin', 'Total_Income'], axis=1)


# In[43]:


train.head()


# # Handling the missing Data

# Checking the number of null values

# In[44]:


train.isnull().sum()


# There are null values in Gender,Married,Dependents,Self_Employed,LoanAmount,Loan_Amount_Term.
# So replacing the null values with the mode of the respective colums so that the values does not affect the result.

# In[45]:


train['Gender'].fillna(train['Gender'].mode()[0],inplace=True)


# In[46]:


train['Married'].fillna(train['Married'].mode()[0],inplace=True)


# In[47]:


train['Dependents'].fillna(train['Dependents'].mode()[0],inplace=True)


# In[48]:


train['Self_Employed'].fillna(train['Self_Employed'].mode()[0],inplace=True)


# In[49]:


train['Credit_History'].fillna(train['Credit_History'].mode()[0],inplace=True)


# In[50]:


train['Loan_Amount_Term'].fillna(train['Loan_Amount_Term'].mode()[0],inplace=True)


# In[51]:


train['LoanAmount'].fillna(train['LoanAmount'].median(),inplace=True)


# In[ ]:





# In[52]:


test['Gender'].fillna(test['Gender'].mode()[0],inplace=True)


# In[53]:


test['Married'].fillna(test['Married'].mode()[0],inplace=True)


# In[54]:


test['Dependents'].fillna(test['Dependents'].mode()[0],inplace=True)


# In[55]:


test['Self_Employed'].fillna(test['Self_Employed'].mode()[0],inplace=True)


# In[56]:


test['Credit_History'].fillna(test['Credit_History'].mode()[0],inplace=True)


# In[57]:


test['Loan_Amount_Term'].fillna(test['Loan_Amount_Term'].mode()[0],inplace=True)


# In[58]:


test['LoanAmount'].fillna(test['LoanAmount'].median(),inplace=True)


# # Outlier Treatment

# There are many outliers in the LoanAmount.Doing the log transformation to make the distribution look normal.

# In[59]:



train['LoanAmount_log'] = np.log(train['LoanAmount']) 
train['LoanAmount_log'].hist(bins=20) 


test['LoanAmount_log'] = np.log(test['LoanAmount'])
test['LoanAmount_log'].hist(bins=20)


# In[ ]:





# # Model Building

# Loan_ID won't be used in the analysis.So, dropping the Loan_ID Column

# In[60]:


train=train.drop('Loan_ID',axis=1)
train.head()


# In[61]:


test=test.drop('Loan_ID',axis=1)
test.head()


# In[ ]:





# In[62]:


train=train.drop('Gender',axis=1)
test=test.drop('Gender',axis=1)


# In[ ]:





# In[63]:


train=train.drop('Dependents',axis=1)
test=test.drop('Dependents',axis=1)


# In[ ]:





# In[64]:


train=train.drop('Self_Employed',axis=1)
test=test.drop('Self_Employed',axis=1)


# In[ ]:





# Also dropping the Loan_Status column and storing it in another variable.

# In[65]:


x=train.drop('Loan_Status',axis=1)
x.head()


# In[66]:


y=train['Loan_Status']
y.head()


# In[ ]:





# Creating Dummy Varible

# In[67]:


x=pd.get_dummies(x) 
train=pd.get_dummies(train) 
test=pd.get_dummies(test)


# In[68]:


x.head()


# In[ ]:





# In[ ]:





# # Applying Logistic Regression

# In[69]:


from sklearn.model_selection import train_test_split
x_train, x_cv, y_train, y_cv = train_test_split(x,y, train_size =0.75,random_state=0)


# In[ ]:





# In[70]:


from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import accuracy_score
model = LogisticRegression() 
model.fit(x_train, y_train)


# In[71]:


pred_cv = model.predict(x_cv)


# In[72]:


accuracy_score(y_cv,pred_cv)


# In[73]:


from sklearn.metrics import confusion_matrix
c=confusion_matrix(y_cv,pred_cv)
c


# In[74]:


test.head()


# In[75]:


pred_test = model.predict(test)


# In[ ]:





# In[76]:


submission=pd.read_csv('/Users/ruchishukla/Downloads/test.csv',header=0)


# In[77]:


submission['Loan_Status']=pred_test 
submission['Loan_ID']=test_original['Loan_ID']


# In[78]:


submission.head()


# In[79]:


submission['Loan_Status'].replace(0, 'N',inplace=True) 
submission['Loan_Status'].replace(1, 'Y',inplace=True)


# In[ ]:





# In[80]:


pd.DataFrame(submission, columns=['Loan_ID','Loan_Status']).to_csv('logistic.csv')


# In[ ]:





# In[ ]:





# In[81]:


submission


# # Thank you
