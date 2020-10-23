### Dependencies
Please use **Python 3.7** for this project, and install the following packages:
```bash
pip install pandas matplotlib category_encoders numpy sklearn tensorflow keras
```
tensorflow requires **numpy 1.16.0**:
```bash
pip install --upgrade numpy==1.16.0
```
# FTEC4003 Course Project Task 1: Insurance Selling

## 1. Background
- This data comes from clients of an insurance company. These clients have already bought the medical insurance. Nowadays, the company wants to launch a new transportation insurance and to find those who will be interested in this insurance.
## 2. Data Set Information
- The data is related with an insurance selling problem. The clients' information is about clients' basic information and their vehicle's situations.
#### This data set contains two files:
1. insurance-train.csv
	- The training set with 11 input attributes and 1 output attribute (i.e. class attribute)
2. insurance-test.csv
	- The testing set with 11 input attributes. You need to identify the class of each item. 

#### Other files
1. samplesubmission.csv:
	- This is a sample file to show the output format. Wrong format will lead to unkown result.

2. evaluate_1.macOS:
	- This is a command line tool to evaluate your result. We will use F1-measure to measure your result.
	- Usage: Press "command + space" to open spotlight search and type in "terminal", then type in the following command in the terminal. You should replace
```./submission_1.csv``` with your own path to the submission_1.csv.
```bash
./evaluate_1.macOS ./submission_1.csv
```

3. evaluate_1.linux:
	- Usage: Press "ctrl + alt + t" to launch a terminal and input the following command.
```bash
./evaluate_1.linux ./submission_1.csv
```

4. evaluate_1.exe:
	- Usage: Press "command + r" and then type in "cmd" in the dialog box to launch a terminal. Then type in the command:
```bash
./evaluate_1.exe ./submission_1.csv
```

## 3. Goal

- The classification goal is to predict if the client will buy the transportation insurance (i.e, Identify the value of feature 'Response', 1 for yes and 0 otherwise).

## 4. Attribute Information
#### a) Input variables
**clients' basic information**
1. ID: Unique ID of clients (numeric)
2. Gender: Gender of clients (categorical: 'Male', 'Female')
3. Age: Age of clients (numeric)
4. Driving_License: whether the clients have a driving license (categorical: '0', '1')
5. Region_Code: Unique code for the region of the clients (numeric)
6. Previously_Insured: whether the clients have already a transportation insurance (categorical: '0', '1')

**clients' vehicle situations**
7. Vehicle_Age: Age of the Vehicle (string)
8. Vehicle_Damage: whether the vehicle has been damaged (categorical: 'No', 'Yes')

**other attributes**
9. Annual_Premium: The amount customer needs to pay as premium in the year (numeric)
10. Policy_Sales_Channel: Anonymised Code for the channel of outreaching to the customer ie. Different Agents, Over Mail, Over Phone, In Person, etc. Here, using unique numbers to represent different channel (numeric)
11. Vintage: Number of Days that Customer has been associated with the company (numeric)

#### b) Output variable
12. Response: whether the client is interested in it (categorical: '0', '1')
# FTEC4003 Course Project Task 2: How many Customers Stay
## 1. Background
- This data comes from clients of a bank. These clients have already had accounts in this bank. Nowadays, the bank wants to model whether they will stay or not in the future. The task is to do the binary classification based on the given information, which gives extra information to the bank to stabilize the customers.
## 2. Data Set Information
- The data are attributes of customers' basic information. 
- train.csv
  - The training set with 13 input attributes and 1 output attribute (i.e. class attribute)
- assignment-test.csv
  - The testing set with 13 input attributes. You need to identify the class of each item. 
#### This data set contains two files:
1. train.csv
	- The training set with known labels
2. assignment-test.csv
	- The testing set without labels (the "Exited" Attribute).

#### Other files
1. samplesubmission.csv:
	
- This is a sample file to show the output format. Wrong format will lead to unknown result.
	
2. evaluate_2.macOS:
	- This is a command line tool to evaluate your result. We will use F1-measure to measure your result. 
	
  - Usage: Press "command + space" to open spotlight search and type in "terminal", then type in the following command in the terminal. You should replace
	```./submission_2.csv``` with your own path to the submission_2.csv.  Please note that ```./```denote the current position of the command line and ```submission_2.csv``` denote your submission file name.
```bash
./evaluate_2.macOS ./submission_2.csv
```

3. evaluate_2.linux:
	- Usage: Press "ctrl + alt + t" to launch a terminal and input the following command.
	- Other  notification details are as introduced in the "macOS setting".
```bash
./evaluate_2.linux ./submission_2.csv
```

4. evaluate_2.exe:
  - Usage: Press "command + r" and then type in "cmd" in the dialog box to launch a terminal. Then type in the command:
  - Other notification details are as introduced in the "macOS setting".
```bash
./evaluate_2.exe ./submission_2.csv
```

## 3. Goal

- The classification goal is to predict if the customer will leave this bank and choose other competitors in the future (i.e, Identify the value of feature 'Exited', 1 for yes and 0 otherwise).

## 4. Attribute Information
#### a) Input variables

**customers' basic information**

- RowNumber: the number of rows
- CustomerId: the id of the customer in this bank.
- Surname: the surname of the customer
- CreditScore: personal credit score for an account.
- Geography: the location of the customer.
- Gender: the gender of the customer.
- Age: the age of the customer.
- Tenure: the valid time of the account.
- Balance: the amount of money in the account.
- NumOfProducts: the number of products the customer buys.
- HasCrCard: The number of Credit Card the customer owns.
- IsActiveMember: whether active in the recent period.
- EstimatedSalary: the estimated salary of the custome

#### b) Output variable

- Exited: whether this customer will leave in the future.

