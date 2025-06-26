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

data = pd.read_csv(filename)
model = LinearRegression()
model.fit(data[[x_col]], data[[y_col]])

plt.scatter(data[[x_col]], data[[y_col]], color='red')
plt.plot(data[[x_col]], model.predict(data[[x_col]]), color='blue')
plt.title(f'{y_col} vs {x_col}')
plt.xlabel(x_col)
plt.ylabel(y_col)
plt.savefig("linear_regression_python_output.png")
plt.show()

#!/usr/bin/env python
# coding: utf-8

# This notebook demonstrates a simple linear regression analysis using Python to model Salary based on Years of Experience.

# In[44]:


import pandas as pd
dataset = pd.read_csv("regression_data.csv")


# In[46]:


import pandas as pd


# In[47]:


dataset = pd.read_csv("regression_data.csv")


# In[48]:


import matplotlib.pyplot as plt


# In[50]:


import matplotlib.pyplot as plt


# In[51]:


plt.scatter(dataset["YearsExperience"], dataset["Salary"], color="red")


# In[52]:


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(dataset[["YearsExperience"]], dataset[["Salary"]])


# In[53]:


import sklearn


# In[55]:


from sklearn.linear_model import LinearRegression


# In[56]:


model = LinearRegression()
model.fit(dataset[["YearsExperience"]], dataset[["Salary"]])


# In[57]:


plt.plot(dataset["YearsExperience"], reg.predict(dataset[["YearsExperience"]]), color="blue")
plt.title("Salary vs Experience")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()


# In[58]:


reg.score(dataset[["YearsExperience"]], dataset[["Salary"]])
# R-squared

