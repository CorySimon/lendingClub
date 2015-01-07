import pandas as pd
import numpy as np
import glob, sys
from datetime import date
import pickle

##Load Data
files = glob.glob('./data/*.csv')
list1 = []
for fileName in files:
    print "Opening file %s" %fileName
    tempFrame = pd.read_csv(fileName, header=1)
    list1.append(tempFrame)

print "Concatenating files..."
loanData = pd.concat(list1, ignore_index=True)


loanData = loanData.drop(["member_id", 
						  "id", "url", 
						  "funded_amnt_inv", 
						  "out_prncp", 
						  "out_prncp_inv", 
						  "total_pymnt_inv", 
						  "total_rec_prncp", 
						  "total_rec_int", 
						  "total_rec_late_fee", 
						  "recoveries", 
						  "collection_recovery_fee", 
						  "desc", "last_pymnt_d", 
						  "last_pymnt_amnt", 
						  "funded_amnt", 
						  "sub_grade", 
						  "emp_title", 
						  "title", 
						  "next_pymnt_d", 
						  "last_credit_pull_d"], 1)

##int_rate
print "Cleaning up int_rate"
loanData['int_rate'] = loanData['int_rate'].str.replace('%', '')
loanData['int_rate'] = loanData['int_rate'].astype(float)


##term
print "Binarizing term"
loanData['term'] = [0 if x == " 36 months" else 1 for x in loanData['term']]

##grade
print "Binarizing grade"
loanData = loanData[pd.isnull(loanData['grade']) == 0]
loanData.index = range(len(loanData))

#grades = dict(zip(loanData['grade'].unique(), np.arange(loanData['grade'].nunique())))
#loanData['grade'] = loanData['grade'].map(lambda x: grades[x])

#Binarize the grade
gradesBinarized = pd.get_dummies(loanData['grade'])
loanData = pd.concat([loanData, gradesBinarized], axis=1)
loanData = loanData.drop(['grade'], 1)


##emp_length
print "Cleaning up emp_years"
emp_years = dict(zip(loanData['emp_length'].unique(), np.arange(loanData['emp_length'].nunique())))
loanData['emp_length'] = loanData['emp_length'].map(lambda x: emp_years[x])


##home_ownership
#loanData['home_ownership'] = ["None" if pd.isnull(x) else x for x in loanData['home_ownership']]
#ownership_dict = dict(zip(loanData['home_ownership'].unique(), np.arange(loanData['home_ownership'].nunique())))
#loanData['home_ownership'] = loanData['home_ownership'].map(lambda x: ownership_dict[x])

homeOwnershipBinarized = pd.get_dummies(loanData['home_ownership'])
loanData = pd.concat([loanData, homeOwnershipBinarized], axis=1)
loanData = loanData.drop(['home_ownership'], 1)

##is_inc_v
print "Binarizing is_inc_v"
loanData['is_inc_v'] = [1 if x == "Source Verified" else 0 for x in loanData['is_inc_v']]


##issue_month and issue_year
loanData['issue_month'] = 0
loanData['issue_year'] = 0

print "Modifying issue_month"
##pd.to_datetime is super slow##
loanData['issue_month'] = [pd.to_datetime(x).month for x in loanData['issue_d']]
print "Modifying issue_year"
loanData['issue_year'] = [pd.to_datetime(x).year for x in loanData['issue_d']]

# for i in range(len(loanData['issue_d'])):
#     m = pd.to_datetime(loanData['issue_d'][i]).month
#     y = pd.to_datetime(loanData['issue_d'][i]).year
#     loanData['issue_month'][i] = m
#     loanData['issue_year'][i] = y
print "Dropping issue_d"
loanData = loanData.drop('issue_d', 1)


##pymnt_plan
print "Binarizing pymnt_plan"
loanData['pymnt_plan'] = [1 if x == 'y' else 0 for x in loanData['pymnt_plan']]


##purpose
# purposes = pd.unique(loanData['purpose'])
# for i in range(len(loanData['purpose'])):
#     for j in range(len(purposes)):
#         if loanData['purpose'][i] == purposes[j]:
#             loanData['purpose'][i] = j  
purposeBinarized = pd.get_dummies(loanData['purpose'])
loanData = pd.concat([loanData, purposeBinarized], axis=1)
loanData = loanData.drop('purpose', 1)

##zip_code
#loanData['zip_code'] = [x[:3] for x in loanData['zip_code']]
loanData = loanData.drop('zip_code', 1)


print "Binarizing addr_state"
##addr_state
#states_dict = dict(zip(loanData['addr_state'].unique(), np.arange(loanData['addr_state'].nunique())))
#loanData['state_id'] = loanData['addr_state'].map(lambda x: states_dict[x])
statesBinarized = pd.get_dummies(loanData['addr_state'])
loanData = pd.concat([loanData, statesBinarized], axis=1)
loanData = loanData.drop('addr_state', 1)


##yrs_since_first_cr_line
print "Cleaning up yrs_since_first_cr_line"
loanData['yrs_since_first_cr_line'] = 0
loanData['yrs_since_first_cr_line'] = [date.today().year-pd.to_datetime(x).year for x in loanData['earliest_cr_line']]

# for i in range(len(loanData['earliest_cr_line'])):
#     earliest_year = pd.to_datetime(loanData['earliest_cr_line'][i]).year
#     loanData['yrs_since_first_cr_line'][i] = date.today().year - earliest_year
loanData = loanData.drop('earliest_cr_line', 1)


##delinquencies
print "Dropping null mths_since_last_delinq"
loanData['mths_since_last_delinq'] = [-1 if pd.isnull(x) else x for x in loanData['mths_since_last_delinq']]


##records
print "Dropping null mths_since_last_record"
loanData['mths_since_last_record'] = [-1 if pd.isnull(x) else x for x in loanData['mths_since_last_record']]


##revol_util
print "Cleaning up revol_util"
loanData['revol_util'] = loanData['revol_util'].str.replace('%', '')
loanData['revol_util'] = loanData['revol_util'].astype(float)
loanData['revol_util'] = [-1 if pd.isnull(x) else x for x in loanData['revol_util']]


##initial_list_status
print "cleaning up initial_list_status"
loanData['initial_list_status'] = [1 if x in ['w', 'W'] else 0 for x in loanData['initial_list_status']]


##major_derog
print "Dropping nulls from mths_since_last_major_derog"
loanData['mths_since_last_major_derog'] = [-1 if pd.isnull(x) else x for x in loanData['mths_since_last_major_derog']]


##collections_12_mths_ex_med
print "Dropping nulls from collections_12_mths_ex_med"
loanData['collections_12_mths_ex_med'] = [-1 if pd.isnull(x) else x for x in loanData['collections_12_mths_ex_med']]


##remove all non-finished loans
print "Removing loans that are still in progress"
loanData = loanData[[x in ['Charged Off',
						   'Default', 
						   'Fully Paid', 
						   'Does not meet the credit policy.  Status:Charged Off',
						   'Does not meet the credit policy.  Status:Default',
						   'Does not meet the credit policy.  Status:Fully Paid'
						   ] for x in loanData['loan_status']]]
loanData.index = range(len(loanData))


##Remove features which aren't available for new loan listings
loanData = loanData.drop('total_pymnt', 1)
loanData = loanData.drop('initial_list_status', 1)
loanData = loanData.drop('policy_code', 1)


##Relabel all defaulted loans as charged off and number them
# for i in range(len(loanData['loan_status'])):
# 	if loanData['loan_status'][i] == 'Default':
# 		loanData['loan_status'][i] = 0
# 	elif loanData['loan_status'][i] == 'Charged Off':
# 		loanData['loan_status'][i] = 0
# 	elif loanData['loan_status'][i] == 'Does not meet the credit policy.  Status:Charged Off':
# 		loanData['loan_status'][i] = 0
# 	elif loanData['loan_status'][i] == 'Does not meet the credit policy.  Status:Default':
# 		loanData['loan_status'][i] = 0
# 	else:
# 		loanData['loan_status'][i] = 1

print "Binarizing loan_status"
loanData['loan_status'] = [0 if x in ['Charged Off',
						   			  'Default',
								      'Does not meet the credit policy.  Status:Charged Off',
								      'Does not meet the credit policy.  Status:Default'
						   			 ] else 1 for x in loanData['loan_status'] ]


##Drop loans with missing values
loanData = loanData.dropna()
loanData.index = range(len(loanData))

print "Pickling the dataframe"

##Pickle the dataframe
f = open('data.pickle', 'wb')
pickle.dump(loanData, f)
f.close()


