import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData['Interest.Rate'][0:5]
loansData['Loan.Length'][0:5]
loansData['FICO.Range'][0:5]
loansData['FICO.Score'] = map(lambda x: int(str(x).split("-")[0]), loansData['FICO.Range'])
loansData['FICO.Score'].head()
loansData['FICO.Score'].hist()
loansData['Interest.Rate'] = map(lambda x: float(str(x)[:-1]), loansData['Interest.Rate'])
loansData.boxplot(column='Monthly.Income', return_type="axes")
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal="hist")
loansData.info()

# Monthly Income is not normal
np.log(loansData['Monthly.Income']).hist()

# Try log transform
loansData['Monthly.Income.Log'] = np.log(loansData['Monthly.Income'])

pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10))
intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']

# reshape the variables
y = np.matrix(intrate).transpose()
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

# Put the columns together to create an input matrix
x = np.column_stack([x1, x2])

# Plus an intercept column
X = sm.add_constant(x)

model = sm.OLS(y, X)
f = model.fit()

# R syntax, all pieces
f.summary()

# Or in chunks
print 'Intercept: ', f.params[0]
print 'Coefficients: ', f.params[1:]
print 'P values: ', f.pvalues
print 'R-squared: ', f.rsquared