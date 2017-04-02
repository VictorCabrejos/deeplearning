
# coding: utf-8

# In[8]:

# Our Dependencies:
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib nbagg')


# # Reading the data
# (1) We read the data using the 'read_fwf' function to read the animal DATASET, a table of fixed-width formatted lines into a pandas dataframe object, which is a 2d data structure of rows and columns.
# 
# (2) Our data set containes the average brain and body weight for a number of animal species.
# 
# (3) Once our data is in our dataframe variables, we can easily parse and weed both measurements into two separate variables.
# 
# (4) We'll store our brain measurements in the x_values variable, and the body measurements in the y-values variable
# 
# (5) So if we were to plot this data right now on a standard 2d graph, it would look like this (scattered). 
# 
# (6) And our goal is that given a new animal's body weight, we would be able to predict what its brain size is.
# 

# In[15]:

dataframe = pd.read_fwf('brain_body.txt')
x_values = dataframe[['Brain']]
y_values = dataframe[['Body']]


# # Train Model on Data
# (1) We use scikit-learn linear_model object to intialize our linear regression and store it in the body_reg (body regression) variable.
# 
# (2) Then we can fit our model on our XY value pairs. 
# 

# In[10]:

body_reg = linear_model.LinearRegression()
body_reg.fit(x_values, y_values)


# # Visualize Results
# (1) Now that we have the line of best fit we can plot our XY value pairs on matplolib scatter plot.
# 
# (2) Then plot our regression line by saying "For every x value we have, predict the associated y value, and draw a line that intersects all those points".
# 
# (3) We can then display it using the 'show' function.
# 
# (4) The x-axis represents brain weights, and the y-axis represents body weights.
# 
# (5) Our regression line seems to fit most of the data pretty well, and there seems to be a very strong correlation here between brain weight and body weight.
# 
# (6) As we move along the line given any brain weight, we can also predict the associated body weight.

# In[16]:

plt.scatter(x_values, y_values)
plt.plot(x_values, body_reg.predict(x_values))
plt.show()


# In[ ]:



