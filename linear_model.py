import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

if len(sys.argv) != 4:
    print("Usage: python linear_regression_python.py <filename> <x_column> <y_column>")
    sys.exit(1)

filename = sys.argv[1]
x_col = sys.argv[2]
y_col = sys.argv[3]

data = pd.read_csv("regression_data.csv")
model = LinearRegression()
model.fit(data[[x_col]], data[[y_col]])

plt.scatter(data[[x_col]], data[[y_col]], color='red')
plt.plot(data[[x_col]], model.predict(data[[x_col]]), color='blue')
plt.title(f'{y_col} vs {x_col}')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.savefig("regression_plot_python.png")
plt.show()

#!/usr/bin/env python
# coding: utf-8

# This notebook demonstrates a simple linear regression analysis using Python to model Salary based on Years of Experience.

# In[1]:


pip install pandas


# In[2]:


import pandas as pd # Installing Pandas to allow regression to run


# In[3]:


dataset = pd.read_csv("regression_data.csv") # Letting Py use the csv as "dataset"


# In[4]:


pip install matplotlib


# In[5]:


import matplotlib.pyplot as plt # Installing matplotlib


# In[6]:


plt.scatter(dataset["YearsExperience"], dataset["Salary"], color="red")
# Creating our initial scatterplot


# In[10]:


pip install scikit-learn


# In[11]:


from sklearn.linear_model import LinearRegression # Importing the linearregression function to run the regression
reg = LinearRegression() # Naming the linear regression as "reg"
reg.fit(dataset[["YearsExperience"]], dataset[["Salary"]]) 


# In[13]:


plt.plot(dataset["YearsExperience"], reg.predict(dataset[["YearsExperience"]]), color="blue")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()
# Plotting the regression line


# In[14]:


reg.score(dataset[["YearsExperience"]], dataset[["Salary"]])
# Displaying the R-squared value for our regression


# In[16]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from sklearn.metrics import mean_squared_error

x = np.array(dataset.YearsExperience)
y = np.array(dataset.Salary)

slope, intercept, r_value, p_value, std_err = linregress(x, y)
y_pred = slope * x + intercept
mse = mean_squared_error(y, y_pred)

plt.plot(x, y_pred, 'r-', label='Fitted Line')
plt.text(1.5, max(y) - 1,
         f"y = {slope:.2f}x + {intercept:.2f}\n"
         f"r = {r_value:.2f}\nMSE = {mse:.2f}",
         fontsize=12)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Regression")
plt.legend()
plt.savefig("regression_plot_python.png") #saving the png file of the regression
plt.show()


# In[17]:


print(slope) # Slope
print("This is the slope for our regression model")


# In[18]:


print(intercept)
print("This is the intercept for our regression model")


# In[19]:


print(r_value)
print("This is the r-value for our regression model")


# In[20]:


print(mse)
print("This is the MSE for our regression model")


# In[ ]:




