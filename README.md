# HR Analytics Employee Attrition
A company has been in industry since a long time. 
Their business had been increasing quite well over past, however in recent years, there has been a slowdown in terms of growth because their best and most experienced employees leaving prematurely.  
The management of the firm is not very happy with the company’s best and most experienced employees leaving prematurely. 
Data scientist has to predict the factors that lead to employee attrition

### Domain Understanding
The key to success in any organization is attracting and retaining top talent. As an HR analyst at any company, one of there task is to determine which factors keep employees in company and which prompt others to leave. 
They have to know what factors they can change to prevent the loss of good people.

### Dimensions of the data
Associated Task : Classification
Target Variable : Attrition (Yes \ No)
Source : IBM Watson Analytics
Number of Instances : 1470
Number of Attributes : 35
Missing Values : No
Class imbalance : Yes 
Domain : Human Resources 
Areas Impacted :  Management, Employees

### Prepare the data
#### >> Observe What data are available
### Personal:
Gender,   Age,   MaritalStatus
TotalWorkingYears,   NumCompaniesWorked
Education,   EducationField
### Job related:
JobLevel,   JobRole,   Department
MonthlyIncome,   StockOptionLevel,   PercentSalaryHike
PerformanceRating,   TrainingTimesLastYear,   JobInvolvement
### Career at company:
YearsAtCompany,   YearsInCurrentRole, 
YearsSinceLastPromotion,   YearsWithCurrManager
### Satisfaction:
RelationshipSatisfaction,   EnvironmentSatisfaction,   JobSatisfaction
### Work intensivity:
WorkLifeBalance,   BusinessTravel,   OverTime,   DistanceFromHome

### Data quality report
#### >> basic statistical analysis

### Understand the levels from data
### >>Observe What are the levels available

### Profile Report for dataset
#### >> Observe Missing values and features rejected

### Analysis
#### >> Pre-processing chart

### Descriptive Analysis
#### >> Observe 3 themes in this analysis

### Workforce composition
How loyal are the employees towards the company?
How diverse is the workforce in terms of education?
How does education relate to employee's performance?

### Gender equality
How are genders distributed across the departments of the company?
How are genders distributed across the hierarchy? 
What differences exist in the compensation of men and women at the same position?

### Turnover (Attrition)
What is the relation between employees satisfaction and attrition?
What impact has the length of the career on the turnover?
How are the different variables correlated?
How good are our features for predicting attrition?

### Profile Report for dataset
### Observe MonthlyIncome highly correlated

### Pie chart of workers
#### >> Observe: Proportion of male is higher than female associates

### Employee Age distribution
#### >> Observe Attrition peaks for employees aged 30

### Percentages of the employees across Education levels and corresponding attrition
#### >> Observe Attrition across Education level and corresponding percentage across the total number of employees

### Histogram – Total Working Years
#### >> Observe TotalWorkingYears highest in 5 to 10 Years

### Count Plot
#### >> Observation  Attrition is higher in employees who are single

## Machine Learning Techniques – Classification Models
Naive Bayes
Logistic Regression
Decision Tree rpart
Decision Tree C5.0
Random Forest
SVM

## Conclusion
### The accuracy of Employee Attrition prediction of machine learning algorithms
From all the above models used, Decision Tree model gave slightly better results when evaluating error metrics on Train & Test data.

Decision Tree model gave best results on this dataset and most important evaluation metric here was Kappa keeping in mind that kappa should not be very low as we want to predict employee attrition.

#### Thanks buddy for reading my document.
## Urs... Jagan Mohan Batchala

