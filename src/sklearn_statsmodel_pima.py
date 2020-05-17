################################################################################
# LINEAR REGRESSION IN PYTHON
################################################################################

#-----------#
# LIBRARIES #
#-----------#

import pandas as pd
import os
import statsmodels.formula.api as smf
import statsmodels.api as sm
import matplotlib.pyplot as plt


#-------------------------#
# EXAMPLE REGRESSION LINE #
#-------------------------#

x = [0, 1, 2, 3, 4, 5]
y = [10, 15, 20, 25, 30, 35]

plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.set_xlabel('X')
ax.set_ylabel('Y')

ax.scatter(x, y, color='b')
ax.set_xlabel('x')
ax.set_ylabel('Y')
ax.plot(x[0:6], y[0:6],color='blue', label = "Y = 10 + 5*x")
plt.grid(alpha=.7,linestyle=':')
plt.hlines(y=10, xmin=0, xmax=1, colors='gray', linestyles='--', lw=2)
plt.vlines(ymin=10, ymax=15, x=1, colors='gray', linestyles='--', lw=2)
plt.legend()
plt.savefig('img/simple_lr_example.png')


#-----------#
# LOAD DATA #
#-----------#

diab = pd.read_csv("data/pima_diabetes.csv")
diab = diab.loc[:, ["Age", "Glucose"]]
diab.head()
diab.describe()

# 0 for Glucose seems implausible
diab = diab.loc[diab["Glucose"] != 0, :]

#-------------#
# SCATTERPLOT #
#-------------#

### Correlation
diab[["Age", "Glucose"]].corr()

### Plot
plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.scatter(diab["Age"], diab["Glucose"], color='b', alpha=0.20)
ax.set_xlabel('Age')
ax.set_ylabel('Glucose')
plt.show()
plt.savefig('img/Age_Glucose_scatter.png')


#----------------------------------#
# STATSMODEL OLS                   #
#----------------------------------#

# Min center age
diab["Age"] = diab["Age"] - diab["Age"].min()

### Statsmodel (No Intercept)
smOLS = sm.OLS(diab["Glucose"], diab["Age"]).fit()
smOLS.summary()

plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.scatter(diab["Age"], diab["Glucose"], color='b', alpha=0.20)
ax.plot(diab["Age"], smOLS.predict(), color='black', alpha=0.70, linewidth=2)
ax.set_xlabel('Age - 21')
ax.set_ylabel('Glucose')
plt.ylim(top=diab['Glucose'].max()+10)
plt.savefig("img/lm_statsmodel_noint.png")

### Statsmodel (With and Without Intercept)
smOLS_int = sm.OLS(diab["Glucose"], sm.add_constant(diab["Age"])).fit()
smOLS_int.summary()

plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.scatter(diab["Age"], diab["Glucose"], color='blue', alpha=0.20)
ax.plot(diab["Age"], smOLS_int.predict(), color='black', label="intercept", alpha=0.70, linewidth=2)
ax.plot(diab["Age"], smOLS.predict(), color='black', label="no intercept",  linestyle=":", alpha=0.50, linewidth=1.5)
ax.set_xlabel('Age - 21')
ax.set_ylabel('Glucose')
ax.legend(loc='lower center', bbox_to_anchor=(0.5, 1.00), shadow=False, ncol=2)
plt.ylim(top=diab['Glucose'].max()+10)
plt.show()
plt.savefig("img/lm_statsmodel_int")


#------------------#
# SCIKIT-LEARN OLS #
#------------------#

from sklearn.linear_model import LinearRegression

sklOLS = LinearRegression().fit(diab["Age"], diab["Glucose"])
X = diab["Age"].to_numpy().reshape(-1, 1)
sklOLS = LinearRegression().fit(X, diab["Glucose"])

sklOLS.intercept_
sklOLS.coef_
sklOLS.score(X, diab["Glucose"])

plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.scatter(diab["Age"], diab["Glucose"], color='b', alpha=0.20)
ax.plot(diab["Age"], sklOLS.predict(X), color='black', alpha=0.70, linewidth=2)
ax.set_xlabel('Age')
ax.set_ylabel('Glucose')
plt.savefig('img/lm_sklearn.png')


#--------------------------#
# TRAINING AND TESTING OLS #
#--------------------------#

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
   X,
   diab["Glucose"],
   test_size= 0.2,
   random_state=0)

sklOLS_train = LinearRegression().fit(X_train, y_train)
sklOLS_train.score(X_train, y_train)
sklOLS_train.score(X_test, y_test)

plt.figure(figsize=(5, 5))
ax = plt.axes()
ax.set_xlabel('predicted')
ax.set_ylabel('actual')
ax.scatter(skOLS_train.predict(X_test), y_test, color='b')
plt.show()
