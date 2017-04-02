
# coding: utf-8

# In[1]:

# Our Dependencies:
#For linear regression model, we use scikit-learn's LinearRegression class.
#This class provides the function fit() to fit the model to your data
import pandas as pd
from sklearn.linear_model import LinearRegression


# # Load the Data
# (1) The data is the file "bmi_and_life_expectancy.csv"
# 
# (2) Using pandas "read_csv" function to load the data into a dataframe
# 
# (3) Assign the dataframe to the variable bmi_life_data

# In[5]:

bmi_life_data = pd.read_csv("bmi_and_life_expectancy.csv")


# # Build a linear regression model
# 
# (1) The model variable is a linear regression model that has been fitted to the data x_values and y_values. 
# 
# (2) Fitting the model means "finding the best line that fits the training data". 
# 
# (3) Create a regression model using scikit-learn's LinearRegression and assign it to "bmi_life_model".
# 
# (4) Fit the model to the data ("bmi_life_data").
# 
# (5) Our goal is to Predict "Life expectancy" (y_value), using a BMI (x_value) 
# 
# (3) Let's make two predictions using the model's predict() function.

# In[6]:

bmi_life_model = LinearRegression()
bmi_life_model.fit(bmi_life_data[['BMI']], bmi_life_data[['Life expectancy']])


# # Predict using the model
# (1) Predict using a BMI of 21.07931 and assign it to the variable "laos_life_exp".

# In[7]:

laos_life_exp = bmi_life_model.predict(21.07931)
print(laos_life_exp)


# In[ ]:




# In[ ]:



