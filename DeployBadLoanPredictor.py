import h2o
from h2o.estimators import H2OGradientBoostingEstimator

h2o.init(nthreads=1, max_mem_size="1500m")

# This is the deployable function

def approve_loan(Loan_Amount, Term, Interest_Rate, Employment_Years, Home_Ownership, Annual_Income, Verification_Status, Loan_Purpose, State,
                 Debt_to_Income, Delinquent_2yr, Revolving_Cr_Util, Total_Accounts, Longest_Credit_Length):
    h2o.connect()
    h2o.remove_all()
    
    loan_application = h2o.H2OFrame({
        'Loan_Amount': [Loan_Amount],
        'Term': [Term],
        'Interest_Rate': [Interest_Rate],
        'Employment_Years': [Employment_Years],
        'Home_Ownership': [Home_Ownership],
        'Annual_Income': [Annual_Income],
        'Verification_Status': [Verification_Status],
        'Loan_Purpose': [Loan_Purpose],
        'State': [State],
        'Debt_to_Income': [Debt_to_Income],
        'Delinquent_2yr': [Delinquent_2yr],
        'Revolving_Cr_Util': [Revolving_Cr_Util],
        'Total_Accounts': [Total_Accounts],
        'Longest_Credit_Length': [Longest_Credit_Length]
    })
    
    loan_approver = h2o.load_model(path="LoanApprover.model")
    prediction = loan_approver.predict(loan_application)
    pred = prediction.as_data_frame()
    values = str(pred.iloc[0, 0])
    return values

