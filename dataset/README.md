# Credit Card Default Dataset
This research aimed at the case of customers default payments in Taiwan and compares the predictive accuracy of probability of default among six data mining methods. From the perspective of risk management the binary result of classification will valuable for identifying credible or not credible clients.

# File descriptions

 - *credit_default_train.csv* - the training set
 - *credit_default_test.csv* - the test set
 - *credit_default_sample_submission.csv* - a sample submission file in the correct format

# Data fields

 - *index* - Id unique to a given row (only in test)
 - *limit* - Amount of the given credit (NT dollar): it includes both the individual consumer credit and his/her family (supplementary) credit.
 - *sex* - Gender
 - *education* - Education
 - *status* - Marital status
 - *age* - Age
 - *ps-sep ... ps-apr* - (Payment Status) History of past payment. We tracked the past monthly payment records (from April to September, 2005) as follows: ps-sep = the repayment status in September, 2005; ps-aug = the repayment status in August, 2005; . . .;ps-apr = the repayment status in April, 2005. The measurement scale for the repayment status is: -2 = no consumption; -1 = paid in full; 0 = the use of revolving credit; 1 = payment delay for one month; 2 = payment delay for two months; . . .; 8 = payment delay for eight months; 9 = payment delay for nine months and above.
 - *ba-sep ... ba-apr* - (Bill Amount) Amount of bill statement (NT dollar). ba-sep = amount of bill statement in September, 2005; ba-aug = amount of bill statement in August, 2005; . . .; ba-apr = amount of bill statement in April, 2005.
 - *pa-sep ... pa-apr* - (Payment Amount) Amount of previous payment (NT dollar). pa-sep = amount paid in September, 2005; pa-aug = amount paid in August, 2005; . . .;pa-apr = amount paid in April, 2005.
