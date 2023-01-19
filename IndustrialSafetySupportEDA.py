#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Importing the data and files
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import seaborn as sns
from scipy import stats; from scipy.stats import zscore, norm, randint

import warnings
warnings.filterwarnings("ignore")


# In[3]:


data= pd.read_csv('IHMStefanini_industrial_safety_and_health_database_with_accidents_description.csv')
print("Shape of the dataset is :",data.shape)
data.head()


# ##### Dropping the unnamed column

# In[4]:


data.drop("Unnamed: 0", axis=1, inplace=True)


# ##### NullValueCheck

# In[5]:


data.isnull().sum()


# ##### Duplicate value Check and Deletion

# In[6]:


print("Shape of the dataset before duplicates deletion is :",data.shape)
print('Number of duplicates in the dataset :',data.duplicated().sum())
data.drop_duplicates(inplace=True)
print("Shape of the dataset after duplicates deletion is :",data.shape)


# In[7]:


print('********Checking the dtypes*********\n')
print(data.dtypes)
print('----------------------------------------------')
print('\n *********Checking the data info********* \n')
print(data.info())


# From the above, it is clearly evident that all the columns of the data frame are of the type object.

# ##### Checking the Statistical Description

# In[8]:


data.describe()


# In[9]:


#Renaming the data, countries,genre, employee or third party to date, country, gender and nature of employee
data.rename(columns={'Data':'Date', 'Countries':'Country', 'Genre':'Gender', 'Employee or Third Party':'Natureofemployee'}, inplace=True)


# In[10]:


data.head(3)


# #### Finding the unique values of all the columns of the dataframe

# In[11]:


col = data[data.columns[~data.columns.isin(['Date','Description'])]].columns.tolist()
for cols in col:
    print(f'Unique values for {cols} is \n{data[cols].unique()}\n')


# #### Replacing and Labelling the values of the columns

# In[12]:


replace_val = {'Local_01': 1, 'Local_02': 2, 'Local_03': 3, 'Local_04': 4, 'Local_05': 5, 'Local_06': 6, 'Local_07': 7, 'Local_08': 8, 'Local_09': 9, 'Local_10': 10, 'Local_11': 11, 'Local_12': 12}
data['Local'] = data['Local'].map(replace_val)
# replace_val = {'I': 1, 'II': 2, 'III': 3, 'IV': 4, 'V': 5}
# data['Accident Level'] = data['Accident Level'].map(replace_val)
# replace_val = {'I': 0, 'II': 1, 'III': 2, 'IV': 3, 'V': 4, 'VI': 5}
# data['Potential Accident Level'] = data['Potential Accident Level'].map(replace_val)
del replace_val


# In[13]:


data.head(5)


# #### Changing the date column to date format

# In[14]:


data['Date'] = pd.to_datetime(data['Date'])


# #### Extracting the month, year and day column to the data.

# In[15]:


data['Year'] = data['Date'].apply(lambda x : x.year)
data['Month'] = data['Date'].apply(lambda x : x.month)
data['Weekday'] = data['Date'].apply(lambda x : x.day_name())


# In[16]:


data.head()


# #### Function to create month into seasons

# In[17]:


# function to create month into seasons
def convert_to_season(x):
    if x in [9, 10, 11]:
        season = 'Spring'
    elif x in [12, 1, 2]:
        season = 'Summer'
    elif x in [3, 4, 5]:
        season = 'Autumn'
    elif x in [6, 7, 8]:
        season = 'Winter'
    return season

data['Season'] = data['Month'].apply(convert_to_season)
data.head(3)


# #### Univariate analysis covering Count, Pie, Scatter, Histogram plots

# In[18]:


features=['Country', 'Local', 'Industry Sector', 'Accident Level',
       'Potential Accident Level', 'Gender', 'Natureofemployee',
       'Critical Risk', 'Year', 'Month', 'Weekday', 'Season']


# In[19]:


def univariate_analysis_categorical(dataset,feature):    
    print("\n")
    print("===========================================================================================")
    print("Univariate Analysis of feature: ",feature)
    print("===========================================================================================\n")
        
           
    print("Unique values: ",feature)
    print("-----------------")        
    print(dataset[feature].unique()) 
        
        
    print("\n")  
    print("-----------------")
    print("Countplot  for feature: ",feature)
    print("-----------------")
        
    plt.figure(figsize=(10,6))
    sns.countplot(dataset[feature],order = dataset[feature].value_counts().index)
    plt.xticks(rotation = 'vertical')
    plt.show()
    
    print("-----------------")
    print("Pie Chart for feature: ",feature)
    print("------------------")      
        
    labels=dataset[feature].unique()
    plt.figure(figsize=(10,6))
    dataset[feature].value_counts().plot.pie(autopct="%.1f%%")
    plt.show()
        
    print("\n")
    print("-----------------")
    print("Histplot  for feature: ",feature)
    print("-------------------")
        
    plt.figure(figsize=(10,6))
    sns.histplot(dataset[feature])
    plt.show()      
    
    print("\n")
    print("-----------------")
    print("Value Counts for feature: ",feature)
    print("-------------------")
    
    print(dataset[feature].value_counts().sort_values(ascending=False))


# In[20]:


#!pip install pyqt5


# #### 1.Country

# In[21]:


univariate_analysis_categorical(data,'Country')


# ----- From the above plots, we can conclude the following
# 
# 1. The country_01 has a count of about 248. Country _02 has a count of about 129. Country_03 has a count of about 41.
# 
# 2. From the above pie chart, it can be infered that the country _01 is the most affected country with about 59% accidents and country_03 is the least affected country.
# 
# 3. From the above output, the country_01 has maximum accidents and country_03 has minimum accidents.

# ##### 2.Local

# In[22]:


#Count plot
univariate_analysis_categorical(data,'Local')


# 1. Most of the accidents happen in Local_03 with a count of about 89.
# 2. Maximum accidents are taken place in local_03 with 21.18% and least accidents are taken place in local_09 and local_11 with 0.47%.
# 3. From the above output, it can be infered that the local_03 is more prone to accidents.
# 4. From the above histogram, it can be observed that the number of accidents in Local_03 are about 90.

# ##### 3.Industry sector

# In[23]:


univariate_analysis_categorical(data,'Industry Sector')


# 1. From the above, it is evident that the mining is prone to more accidents with about 237.
# 2. It can be observed that the mostly affected sector is Mining sector. 56.71% of accidents occur in Mining sector.
# 3. It is clearly evident that the mining is prone to more accidents

# ##### 4.Accident level

# In[24]:


univariate_analysis_categorical(data,'Accident Level')


# 1. From the above count plot, it is clearly evident that the most accidents belongs to "Accident Level" "1" with a count of about more than 300.
# 2. From the above pie chart, it can be determined that the maximum accidents are of level 1 equivalent to about 73.9%
# 

# ##### 5.potential accident level

# In[25]:


univariate_analysis_categorical(data,'Potential Accident Level')


# 1. From the above count plot, it can be determined that the most "Potential Accident Level" belongs to level IV with a count of about 141.
# 2. From the above pie chart, it is evident that most "Potential Accident Level" belongs to level IV with 33.7%.

# ##### 6.Gender

# In[26]:


univariate_analysis_categorical(data,'Gender')


# 1. From the above plot, it is evident that Most affected wokers in accidents are male with a count of 396.
# 2. From the above pie chart, it is evient that most affected wokers in accidents are male.

# ##### 7.Nature of the employee

# In[27]:


univariate_analysis_categorical(data,'Natureofemployee')


# From the above it can be determined that the employee type of Third party are prone to accidents.

# ###### 8.Critical risk

# In[28]:


#Count plot
# plt.figure(figsize=(20,5))
# descending_order = data['Critical Risk'].value_counts().sort_values(ascending=False).index
# sns.countplot(x=data['Critical Risk'],order=descending_order);
# plt.xticks(rotation = 'vertical')

univariate_analysis_categorical(data,'Critical Risk')


# When we count the number of incidents by each type of critical risk, Others tops the list.

# ###### 9.Year

# In[29]:


univariate_analysis_categorical(data,'Year')


# From the above, it is clearly evident that most accidents happend in year 2016. i.e- more than 250.

# ##### 10.Month

# In[30]:


univariate_analysis_categorical(data,'Month')


# 1. From the above, it can be determined that the most accidents are of month Feb.
# 2. From the above, it is evident that most of the accidents happened in feb equivalent to 14.6%.

# ##### 11.WeekDay

# In[31]:


univariate_analysis_categorical(data,'Weekday')


# 1. From the above count plot, it can be determined that max accidents happened on thursday with approximately 76 accidents.
# 2. From the above piechart, it is evident that most accidents happend in Thursday equivalent to 18.2%

# ### Bivariate analysis

# ### Gender vs RestAll

# ##### 1.Gender vs Accident level

# In[32]:


sns.countplot(x="Accident Level",hue="Gender", data=data)


# In[33]:


bivariate_analysis_df = pd.crosstab(index=data['Accident Level'],columns=data['Gender'])

print("------------------------------------------")
print("Cross table Analysis of features: ",'Accident Level',' and  ', 'Gender')
print("------------------------------------------")
display(bivariate_analysis_df)  


# From the above count plot, it can be determined that the most of the accidents happened at level I with gender male.

# #### 2.Gender vs Potential Accident Level

# In[34]:


sns.countplot(x="Potential Accident Level",hue="Gender", data=data)


# In[35]:


bivariate_analysis_df = pd.crosstab(index=data['Potential Accident Level'],columns=data['Gender'])

print("\n Cross table Analysis of features: ",'Potential Accident Level',' and  ', 'Gender')
print("--------------------------------------------------------------------------")
display(bivariate_analysis_df)  


# From the above,it can be determined that most of the potential level accidents happened to male compared to female, of which Potential Accident Level of IV is dominant

# #### 3.Gender vs Country

# In[36]:


sns.countplot(x="Country",hue="Gender", data=data)


# In[37]:


bivariate_analysis_df = pd.crosstab(index=data['Country'],columns=data['Gender'])

print("\n Cross table Analysis of features: ",'Country',' and  ', 'Gender')
print("------------------------------------------------------")
display(bivariate_analysis_df)  


# From the above countplot, it can be determined that the maximum number of accidents took place in country_01 to males and they are about 241.

# ##### 4.Gender vs Industry Sector

# In[38]:


#count plot to determine the number of accidents happened due to industry sector with their gender
sns.countplot(x="Industry Sector",hue="Gender", data=data)


# In[39]:


bivariate_analysis_df = pd.crosstab(index=data['Industry Sector'],columns=data['Gender'])

print("\n Cross table Analysis of features: ",'Industry Sector',' and  ', 'Gender')
print("------------------------------------------------------")
display(bivariate_analysis_df)  


# From the above count plot, it is evident that most of the accidents happened to Male in the mining sector, around 232.

# ##### 5.Gender vs Year

# In[40]:


#Countplot to find in which max accidents took place to both female and male
sns.countplot(x="Year",hue="Gender", data=data)


# In[41]:


bivariate_analysis_df = pd.crosstab(index=data['Year'],columns=data['Gender'])

print("\n Cross table Analysis of features: ",'Year',' and  ', 'Gender')
print("------------------------------------------------------")
display(bivariate_analysis_df)  


# From the above countplot, it is clearly evident that maximum accidents took place in 2016 to the male when compared to female with a count of 269.

# #### 6.Gender vs month

# In[42]:


#Countplot to determine in which month the maximum accidents took place to both female and males
sns.countplot(x="Month",hue="Gender", data=data)


# In[43]:


bivariate_analysis_df = pd.crosstab(index=data['Month'],columns=data['Gender'])

print("\n Cross table Analysis of features: ",'Month',' and  ', 'Gender')
print("------------------------------------------------------")
display(bivariate_analysis_df)  


# From the above count plot, it is determined that maximum number of accidents happened to male in the month feb with a count 57.

# ##### 7.Gender vs weekday

# In[44]:


#Countplot to find out on which day max accidents took place to the both genders
sns.countplot(x="Weekday",hue="Gender", data=data)


# In[45]:


bivariate_analysis_df = pd.crosstab(index=data['Weekday'],columns=data['Gender'])

print("\n Cross table Analysis of features: ",'Weekday',' and  ', 'Gender')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Max accidents happened to male on thursday with a count of more than 73

# ##### 8. Gender vs Nature of Employee

# In[46]:


#count plot to determine which type of employee and gender faced most of the accidents
sns.countplot(x="Natureofemployee",hue="Gender", data=data)


# In[47]:


bivariate_analysis_df = pd.crosstab(index=data['Natureofemployee'],columns=data['Gender'])

print("\n Cross table Analysis of features: ",'Natureofemployee',' and  ', 'Gender')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# From the above output, it is clearly evident that maximum accidents happened to third party male employees. i.e- 176.

# ##### 9. Gender vs Critical Risk

# In[48]:


#count plot to determine which type of employee and gender faced most of the accidents
sns.countplot(x="Critical Risk",hue="Gender", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)
plt.xticks(rotation=90)


# In[49]:


bivariate_analysis_df = pd.crosstab(index=data['Critical Risk'],columns=data['Gender'])

print("\n Cross table Analysis of features: ",'Critical Risk',' and  ', 'Gender')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Critical Risk of type "Others" is dominant across both Male and Female Genders

# ### Industry sector vs RestAll

# ##### 1. Industry Sector Vs Accident Level

# In[50]:


#Countplot to determine the accident level happened at different industry sectors
sns.countplot(x="Industry Sector",hue="Accident Level", data=data)


# In[51]:


bivariate_analysis_df = pd.crosstab(index=data['Industry Sector'],columns=data['Accident Level'])

print("\n Cross table Analysis of features: ",'Industry Sector',' and  ', 'Accident Level')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Maximum number of accidents happened in the mining sector with accident Level I. i.e- 163.

# ##### 2.Industry sector vs potential accident level

# In[52]:


#Countplot to determine the potential accident level according to the industry sector
sns.countplot(x="Industry Sector",hue="Potential Accident Level", data=data)


# In[53]:


bivariate_analysis_df = pd.crosstab(index=data['Industry Sector'],columns=data['Potential Accident Level'])

print("\n Cross table Analysis of features: ",'Industry Sector',' and  ', 'Potential Accident Level')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Maximum number of accidents happened in the potential accident level 4 and mining sector with a count 99. Minimum number of accidents took place in the mining sector at a potential accident level 6.

# ##### 3.Industry Sector vs Critical Risk

# In[54]:


#Countplot to determine the number of accidents taken place at the industry sector wrt critical risk
fig = plt.figure(figsize = (15, 7.2))
ax = fig.add_subplot(121)
sns.countplot(x = 'Critical Risk', data = data, ax = ax, orient = 'v',
                  hue = 'Industry Sector')
plt.legend(labels = data['Industry Sector'].unique())
plt.xticks(rotation = 90)


# In[55]:


bivariate_analysis_df = pd.crosstab(index=data['Critical Risk'],columns=data['Industry Sector'])

print("\n Cross table Analysis of features: ",'Critical Risk',' and  ', 'Industry Sector')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# From the above count plot, it is evident that maximum number of accidents happened in mining with a critical risk of others. i.e- about 175

# ##### 4.Industry sector vs Local

# In[56]:


sns.countplot(x="Local",hue="Industry Sector", data=data)


# In[57]:


bivariate_analysis_df = pd.crosstab(index=data['Industry Sector'],columns=data['Local'])

print("\n Cross table Analysis of features: ",'Industry Sector',' and  ', 'Local')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Many accidents happened with a local 3 and industrial sector mining. i.e- more than 80. Least accidents took place with local 11 and industrial sector others.

# ##### 5. Industry Sector Vs Year

# In[58]:


#Count plot to determine the number of accidents taken place in year 2016 and 2017 according to the industrial sector
sns.countplot(x="Year",hue="Industry Sector", data=data)


# In[59]:


bivariate_analysis_df = pd.crosstab(index=data['Industry Sector'],columns=data['Year'])

print("\n Cross table Analysis of features: ",'Industry Sector',' and  ', 'Year')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# #### From the above plot, the following could be determined
# 
# 1.The number of accidents taken place in year 2016 for mining sector is 160.
# 
# 2.The number of accidents taken place in year 2016 wrt metals sector is about 100.
# 
# 3.The number of accidents taken place in the year 2016 wrt others sector is about 30.
# Hence, it can be determined that maximum accidents took place in mining sector in the year 2016.
# 
# 4.The number of accidents taken place in the year 2017 wrt mining sector is 80.
# 
# 5.The number of accidents taken place in the year 2017 wrt metals sector is about 40.
# 
# 6.The number of accidents taken place in the year 2017 wrt others sector is 20.
# 
# Hence, it can be determined that max accidents took place in mining sector in the year 2017

# ##### 6. Industry Sector Vs Month

# In[60]:


#Count plot to determine the accidents taken place in all the months wrt industrial sector
sns.countplot(x="Industry Sector",hue="Month", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[61]:


bivariate_analysis_df = pd.crosstab(index=data['Industry Sector'],columns=data['Month'])

print("\n Cross table Analysis of features: ",'Industry Sector',' and  ', 'Month')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Maximum number of accidents happened in the month feb and mining sector. The least number of accidents took place in the others sector and month december.

# ##### 7.Industry sector vs Weekday

# In[62]:


sns.countplot(x="Industry Sector",hue="Weekday", data=data)


# In[63]:


bivariate_analysis_df = pd.crosstab(index=data['Industry Sector'],columns=data['Weekday'])

print("\n Cross table Analysis of features: ",'Industry Sector',' and  ', 'Weekday')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Maximum number of accidents hapenned on the day saturday in the mining sector. i.e- more than 40. The least number of accidents happened on the day sunday in the others sector.

# ##### 8. Industry sector Vs country

# In[64]:


sns.countplot(x="Industry Sector",hue="Country", data=data)


# In[65]:


bivariate_analysis_df = pd.crosstab(index=data['Industry Sector'],columns=data['Country'])

print("\n Cross table Analysis of features: ",'Industry Sector',' and  ', 'Country')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# From the above count plot, it is evident that the maximum number of accidents took place in country_01 and mining sector.i.e- 200. The least number of accidents took place in country _01 and others sector.

# ##### 9.Industry Sector Vs nature of employee

# In[66]:


sns.countplot(x="Industry Sector",hue="Natureofemployee", data=data)


# In[67]:


bivariate_analysis_df = pd.crosstab(index=data['Industry Sector'],columns=data['Natureofemployee'])

print("\n Cross table Analysis of features: ",'Industry Sector',' and  ', 'Natureofemployee')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# From the above count plot, it is clearly evident that the maximum accidents took place in the mining sector with the third party employee type. i.e- about 120. The least number of accidents took place in the others sectors with the nature of employee as employee.

# ### Country vs RestAll

# ##### 1. Country vs Year

# In[68]:


sns.countplot(x="Country",hue="Year", data=data)


# In[69]:


bivariate_analysis_df = pd.crosstab(index=data['Country'],columns=data['Year'])

print("\n Cross table Analysis of features: ",'Country',' and  ', 'Year')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# From the above output, the following can be determined-
# 
# 1.The number of accidents taken place in country_01 and year 2016 is 174.
# 
# 2.The number of accidents taken place in country_01 and year 2017 is about 74.
# 
# 3.The number of accidents taken place in country_02 and year 2016 is more than 86.
# 
# 4.The number of accidents taken place in country_02 and year 2017 is about 43.
# 
# 5.The number of accidents taken place in country_03 and year 2016 is about 23.
# 
# 6.The number of accidents taken place in country_03 and year 2017 is about 18.

# ##### 2. Country Vs accident level

# In[70]:


sns.countplot(x="Country",hue="Accident Level", data=data)


# In[71]:


bivariate_analysis_df = pd.crosstab(index=data['Country'],columns=data['Accident Level'])

print("\n Cross table Analysis of features: ",'Country',' and  ', 'Accident Level')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# From the above count plot, it is clearly evident that the maximum number of accidents took place in accident level 1 and country_01.

# ##### 3. Country Vs Potential Accident Level

# In[72]:


sns.countplot(x="Country",hue="Potential Accident Level", data=data)


# In[73]:


bivariate_analysis_df = pd.crosstab(index=data['Country'],columns=data['Potential Accident Level'])

print("\n Cross table Analysis of features: ",'Country',' and  ', 'Potential Accident Level')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# From the above plot, it is evident that the maximum accidents occurred in country_01 and potential accident level 3.

# #### 4. Country Vs Local

# In[74]:


sns.countplot(x="Country",hue="Local", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[75]:


bivariate_analysis_df = pd.crosstab(index=data['Country'],columns=data['Local'])

print("\n Cross table Analysis of features: ",'Country',' and  ', 'Local')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Country 1 is more dominant in local 3 region and least dominant in Local 12

# ##### 5.Country Vs Nature Of Employee

# In[76]:


sns.countplot(x="Country",hue="Natureofemployee", data=data)


# In[77]:


bivariate_analysis_df = pd.crosstab(index=data['Country'],columns=data['Natureofemployee'])

print("\n Cross table Analysis of features: ",'Country',' and  ', 'Natureofemployee')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Accidents in Country 01 is more dominant in Third Party type of employee, country 03 is least dominant in Third Party (Remote)

# ##### 6. Country Vs Critical Risk

# In[78]:


sns.countplot(x="Country",hue="Critical Risk", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[79]:


bivariate_analysis_df = pd.crosstab(index=data['Critical Risk'],columns=data['Country'])

print("\n Cross table Analysis of features: ",'Country',' and  ', 'Critical Risk')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Country 01 is more dominant in Others Critical Risk and Critical Risk is least dominant in Country 03 

# ### Local Vs Rest All

# ##### 1. Local Vs Accident Level

# In[80]:


sns.countplot(x="Local",hue="Accident Level", data=data)


# In[81]:


bivariate_analysis_df = pd.crosstab(index=data['Accident Level'],columns=data['Local'])

print("\n Cross table Analysis of features: ",'Local',' and  ', 'Accident Level')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Accident level 1 is more dominant in Local 2 region with 65 accidents, while Accident Level V is least across all Locals

# ##### 2. Local Vs Potential Accident Level

# In[82]:


sns.countplot(x="Local",hue="Potential Accident Level", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[83]:


bivariate_analysis_df = pd.crosstab(index=data['Potential Accident Level'],columns=data['Local'])

print("\n Cross table Analysis of features: ",'Local',' and  ', 'Potential Accident Level')
print("------------------------------------------------------")
display(bivariate_analysis_df) 


# Overall Local 3 is more prone to Multiple potential accidents, while local 12 is the least

# ##### 3. Local Vs Natureofemployee

# In[84]:


sns.countplot(x="Local",hue="Natureofemployee", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[85]:


bivariate_analysis_df = pd.crosstab(index=data['Natureofemployee'],columns=data['Local'])

print("\n Cross table Analysis of features: ",'Local',' and  ', 'Natureofemployee')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Type Employee is more dominant across all Locals, while Type Third Party(Remote) is least dominant across all Locals

# ##### 4. Local Vs Critical Risk

# In[86]:


sns.countplot(x="Local",hue="Critical Risk", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[87]:


bivariate_analysis_df = pd.crosstab(index=data['Critical Risk'],columns=data['Local'])

print("\n Cross table Analysis of features: ",'Local',' and  ', 'Critical Risk')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Critical Risk of type "Others" is dominant across all Locals

# ##### 5. Local Vs Year

# In[88]:


sns.countplot(x="Local",hue="Year", data=data)


# In[89]:


bivariate_analysis_df = pd.crosstab(index=data['Year'],columns=data['Local'])

print("\n Cross table Analysis of features: ",'Local',' and  ', 'Year')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Year 2016 has more accidents across all Local regions compared to 2017

# ### Accident Level Vs Rest All

# ##### 1. Accident Level Vs Potential Accident Level

# In[90]:


sns.countplot(x="Accident Level",hue="Potential Accident Level", data=data)


# In[91]:


bivariate_analysis_df = pd.crosstab(index=data['Potential Accident Level'],columns=data['Accident Level'])

print("\n Cross table Analysis of features: ",'Accident Level',' and  ', 'Potential Accident Level')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Accident Level I is more related to Potential Accident levels of I, II, III, IV, V, VI

# ##### 2. Accident Level Vs Natureofemployee

# In[92]:


sns.countplot(x="Accident Level",hue="Natureofemployee", data=data)


# In[93]:


bivariate_analysis_df = pd.crosstab(index=data['Accident Level'],columns=data['Natureofemployee'])

print("\n Cross table Analysis of features: ",'Accident Level',' and  ', 'Natureofemployee')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Accident Level I is more dominant across all Employee types, where Level V is least across all types

# ##### 3. Accident Level Vs Critical Risk

# In[94]:


sns.countplot(x="Accident Level",hue="Critical Risk", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[95]:


bivariate_analysis_df = pd.crosstab(index=data['Critical Risk'],columns=data['Accident Level'])

print("\n Cross table Analysis of features: ",'Accident Level',' and  ', 'Critical Risk')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Accident Level I is more domaint with Other critical Risk type

# ##### 4. Accident Level Vs Year

# In[96]:


sns.countplot(x="Accident Level",hue="Year", data=data)


# In[97]:


bivariate_analysis_df = pd.crosstab(index=data['Year'],columns=data['Accident Level'])

print("\n Cross table Analysis of features: ",'Accident Level',' and  ', 'Year')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Accident Level I is more dominant in across 2016 and 2017 years, and Level V is minimum

# #### 5. Accident Level Vs Month

# In[98]:


sns.countplot(x="Month",hue="Accident Level", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[99]:


bivariate_analysis_df = pd.crosstab(index=data['Accident Level'],columns=data['Month'])

print("\n Cross table Analysis of features: ",'Accident Level',' and  ', 'Month')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Accident Level 1 dominates across all Months while Level V is minimum

# #### 6. Accident Level Vs Country

# In[100]:


sns.countplot(x="Country",hue="Accident Level", data=data)


# In[101]:


bivariate_analysis_df = pd.crosstab(index=data['Accident Level'],columns=data['Country'])

print("\n Cross table Analysis of features: ",'Accident Level',' and  ', 'Country')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Accident Level I is more dominant across all Countries, while Accident Level V is least dominant across all countries

# ### Potential Accident Level Vs Rest All(Remaining uncoverd)

# ##### 1. Potential Accident Level Vs Natureofemployee

# In[102]:


sns.countplot(x="Potential Accident Level",hue="Natureofemployee", data=data)


# In[103]:


bivariate_analysis_df = pd.crosstab(index=data['Potential Accident Level'],columns=data['Natureofemployee'])

print("\n Cross table Analysis of features: ",'Potential Accident Level',' and  ', 'Natureofemployee')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Potential Accident level IV dominents in ThirdParty, while VI is least dominant in Third Party(Remote) across all

# ##### 2. Potential Accident Level Vs Critical Risk

# In[104]:


sns.countplot(x="Potential Accident Level",hue="Critical Risk", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[105]:


bivariate_analysis_df = pd.crosstab(index=data['Critical Risk'],columns=data['Potential Accident Level'])

print("\n Cross table Analysis of features: ",'Potential Accident Level',' and  ', 'Critical Risk')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Among all Critical Risk with Type as "Others" is dominant across all Potential Accident Levels

# In[106]:


plt.figure(figsize=(10,6))
sns.barplot(data['Accident Level'], data['Month'], hue=data['Year'], palette='muted') 


# ##### 3. Potential Accident Level Vs Year

# In[107]:


sns.countplot(x="Potential Accident Level",hue="Year", data=data)


# In[108]:


bivariate_analysis_df = pd.crosstab(index=data['Potential Accident Level'],columns=data['Year'])

print("\n Cross table Analysis of features: ",'Potential Accident Level',' and  ', 'Year')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# There is Decrease in Number of accidents across all Potential Accident level from 2016 to 2017. Potential Accident level IV is dominant in both 2016 and 2017

# #### 4. Potential Accident Level Vs Country

# In[109]:


sns.countplot(x="Potential Accident Level",hue="Country", data=data)


# In[110]:


bivariate_analysis_df = pd.crosstab(index=data['Potential Accident Level'],columns=data['Country'])

print("\n Cross table Analysis of features: ",'Potential Accident Level',' and  ', 'Country')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Potential Accident Level IV is dominant across all countries, while with VI least number of accidents happenned 

# ### Natureofemployee Vs RestAll

# ##### 1. Natureofemployee Vs Critical Risk

# In[111]:


sns.countplot(x="Natureofemployee",hue="Critical Risk", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[112]:


bivariate_analysis_df = pd.crosstab(index=data['Critical Risk'],columns=data['Natureofemployee'])

print("\n Cross table Analysis of features: ",'Critical Risk',' and  ', 'Natureofemployee')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# Critical Risk of type "Others" is dominant across all Types of Employees

# ### Year Vs RestAll

# ##### 1. Year vs month

# In[113]:


sns.countplot(x="Month",hue="Year", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[114]:


bivariate_analysis_df = pd.crosstab(index=data['Month'],columns=data['Year'])

print("\n Cross table Analysis of features: ",'Month',' and  ', 'Year')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# From the above plot, it is evident that the max accidents happened in the year 2016 and march.

# ##### 2. Year vs Local

# In[115]:


sns.countplot(x="Local",hue="Year", data=data)


# In[116]:


bivariate_analysis_df = pd.crosstab(index=data['Year'],columns=data['Local'])

print("\n Cross table Analysis of features: ",'Local',' and  ', 'Year')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# From the above plot, it can be determined that the maximum accidents took place in the local 3 and year 2016.

# ##### 3. Year vs Weekday

# In[117]:


sns.countplot(x="Weekday",hue="Year", data=data)


# In[118]:


bivariate_analysis_df = pd.crosstab(index=data['Year'],columns=data['Weekday'])

print("\n Cross table Analysis of features: ",'Weekday',' and  ', 'Year')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# From the above plot, it is clearly evident that maximum number of accidents took place on thursday and year 2016.

# #### 4.Year vs Critical Risk

# In[119]:


sns.countplot(x="Year",hue="Critical Risk", data=data)
plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left', borderaxespad=0)


# In[120]:


bivariate_analysis_df = pd.crosstab(index=data['Critical Risk'],columns=data['Year'])

print("\n Cross table Analysis of features: ",'Critical Risk',' and  ', 'Year')
print("------------------------------------------------------")
display(bivariate_analysis_df)


# From the above plot, it is clearly evident that maximum number of accidents took place with "Others" and year 2016.

# In[ ]:





# ### Multivariate analysis

# In[121]:


#Pair plot
sns.pairplot(data)


# In[122]:


data.corr()


# From the above Correlation diagram its clear that "Local" and "Year" are moderately correlated

# In[123]:


data.columns


# ### ********Groupby Analysis**********

# #### Year wise distribution of accidents and potential accident levels

# In[124]:


data.groupby(['Year','Accident Level','Potential Accident Level'])[['Accident Level']].count()


# 1. Year 2016 with Accident Level I has maximum accidents of 64 with Potential Accident Level III and 62 with Potential Accident Level II
# 
# 2. Year 2017 with Accident Level I has maximum accidents of 26 with Potential Accident Level II and 25 with Potential Accident Level III,IV

# #### Year wise distribution of Industry Sector and accident levels

# In[125]:


data.groupby(['Year','Industry Sector','Accident Level'])[['Accident Level']].count()


# 1. Year 2016 with Industry Sector of Type "Metals" has maximum accidents of 79 with Accident Level I 
# 2. Year 2016 with Industry Sector of Type "Mining" has maximum accidents of 112 with Accident Level I 
# 3. Year 2016 with Industry Sector of Type "Others" has maximum accidents of 20 with Accident Level I
# 
# 4. Year 2017 with Industry Sector of Type "Metals" has maximum accidents of 28 with Accident Level I 
# 5. Year 2017 with Industry Sector of Type "Mining" has maximum accidents of 51 with Accident Level I 
# 6. Year 2017 with Industry Sector of Type "Others" has maximum accidents of 19 with Accident Level I

# #### Industry Sector wise distribution of Country and accident levels

# In[126]:


data.groupby(['Industry Sector','Country','Accident Level'])[['Accident Level']].count()


# 1. Metals in Country_01 has maximum accidents with Level 1 with 36 count
# 2. Metals in Country_02 has maximum accidents with Level 1 with 71 count
# 3. Mining in Country_01 has maximum accidents with Level 1 with 140 count
# 4. Mining in Country_02 has maximum accidents with Level 1 with 23 count
# 5. Others in Country_03 has maximum accidents with Level 1 with 34 count

# ### Word Cloud Analysis

# In[127]:


#!pip install wordcloud
#!pip install pandas_profiling
from wordcloud import WordCloud


# ### WordCloud for Accident Level and Description

# In[128]:


for i in data['Accident Level'].unique():
    print('WordCloud for Accident Level :', i,'\n')
    text = " ".join(cat.split()[1] for cat in data[data['Accident Level'] == i]['Description'])
    # Creating word_cloud with text as argument in .generate() method
    word_cloud = WordCloud(collocations = False, background_color = 'lightyellow').generate(text)
    # Display the generated Word Cloud
    plt.figure(figsize=[10,10])
    
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show
    print('-----------------------------')
    


# ### WordCloud for Potential Accident Level and Description

# In[129]:


for i in data['Potential Accident Level'].unique():
    if i != 'VI':
        print('WordCloud for Potential Accident Level :', str(i),'\n')
        text = " ".join(cat.split()[1] for cat in data[data['Potential Accident Level'] == i]['Description'])
        # Creating word_cloud with text as argument in .generate() method
        word_cloud = WordCloud(collocations = False, background_color = 'lightyellow').generate(text)
        # Display the generated Word Cloud
        plt.figure(figsize=[10,10])

        plt.imshow(word_cloud, interpolation='bilinear')
        plt.axis("off")
        plt.show
        print('-----------------------------')


# ### WordCloud for Industry Sector and Description

# In[130]:


for i in data['Industry Sector'].unique():
    print('WordCloud for Industry type :', str(i),'\n')
    text = " ".join(cat.split()[1] for cat in data[data['Industry Sector'] == i]['Description'])
    # Creating word_cloud with text as argument in .generate() method
    word_cloud = WordCloud(collocations = False, background_color = 'lightyellow').generate(text)
    # Display the generated Word Cloud
    plt.figure(figsize=[10,10])
    
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show
    print('-----------------------------')


# ### WordCloud for Country and Description

# In[131]:


for i in data['Country'].unique():
    print('WordCloud for Country :', i,'\n')
    text = " ".join(cat.split()[1] for cat in data[data['Country'] == i]['Description'])
    # Creating word_cloud with text as argument in .generate() method
    word_cloud = WordCloud(collocations = False, background_color = 'lightyellow').generate(text)
    # Display the generated Word Cloud
    plt.figure(figsize=[10,10])
    
    plt.imshow(word_cloud, interpolation='bilinear')
    plt.axis("off")
    plt.show
    print('-----------------------------')
    


# In[134]:


get_ipython().system('pip install pandas-profiling')
from pandas_profiling import ProfileReport
profile = ProfileReport(data, title="Pandas Profiling Report")
profile.to_notebook_iframe()


# In[135]:


#Exporting the finalized dataset to csv


# In[137]:


data2=data.to_csv(r'data.csv', index = False)


# In[ ]:




