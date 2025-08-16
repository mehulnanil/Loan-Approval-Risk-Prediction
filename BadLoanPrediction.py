import h2o
from h2o.estimators import H2OGradientBoostingEstimator
import pandas as pd

# Initialize H2O
h2o.init(nthreads=1, max_mem_size="1500m")

# Load some data
customerData = h2o.import_file("https://s3-us-west-1.amazonaws.com/dsclouddata/LendingClubData/Loans-Customer-Info.csv")
accountData = h2o.import_file("https://s3-us-west-1.amazonaws.com/dsclouddata/LendingClubData/Loans-Account-Info.csv")

# View the data
customerData.describe()
customerData.head()

# Join the files into a single frame
customer360 = customerData.merge(accountData, by="RowID")
customer360.head()

# Pick a response variable for the supervised problem
response = "Bad_Loan"

# Use all other columns (except for the name) as predictors
predictors = [col for col in customer360.columns if col not in [response, "RowID", "Interest_Rate"]]

# Split dataset giving the training dataset 75% of the data
customer360_split = customer360.split_frame(ratios=[0.75])

# Create a training set from the 1st dataset in the split
customer360_train = customer360_split[0]

# Create a testing set from the 2nd dataset in the split
customer360_test = customer360_split[1]

# Train the model
LoanApprover = H2OGradientBoostingEstimator(model_id="LoanApprover.model")
LoanApprover.train(x=predictors, y=response, training_frame=customer360_train)

# Display the model
print(LoanApprover)

# Variable importance plot
h2o.varimp_plot(LoanApprover)
# h2o.partial_plot(LoanApprover, data=customer360_train, cols=["Interest_Rate", "Annual_Income"])

# h2o.save_model(LoanApprover, path="Models")

# h2o.predict(LoanApprover, newdata=customer360_test)

