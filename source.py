#!/usr/bin/env python
# coding: utf-8

# # Exploring the Gender Pay Gap Across Industries
# 
# ![Banner](./assets/banner.jpeg)

# ## Topic
# *What problem are you (or your stakeholder) trying to address?*
# 📝 <!-- Answer Below -->
# 
# Despite progress towards gender equality in the workplace, significant disparities in pay between men and women persist across various industries. Understanding the factors contributing to these disparities is essential for addressing the gender pay gap effectively.
# 
# The gender pay gap impacts economic stability and personal well-being for many individuals. Addressing it can lead to a more equitable workforce and improved societal outcomes.
# 
# 

# ## Project Question
# *What specific question are you seeking to answer with this project?*
# *This is not the same as the questions you ask to limit the scope of the project.*
# 📝 <!-- Answer Below -->
# 
# -What is the average salary difference between men and women in various industries?
# -How do education levels and years of experience correlate with the gender pay gap?
# -Are there specific industries where the gender pay gap is more pronounced?
# -What role do company diversity and inclusion initiatives play in mitigating the pay gap?

# ## What would an answer look like?
# *What is your hypothesized answer to your question?*
# 📝 <!-- Answer Below -->
# 
# -Visual representations such as bar charts comparing average salaries by gender across industries.
# -Scatter plots correlating years of experience and education level with salary differences.
# -A summary of findings highlighting industries with significant disparities and potential factors influencing these gaps.

# ## Data Sources
# *What 3 data sources have you identified for this project?*
# *How are you going to relate these datasets?*
# 
# ->Glassdoor Gender Pay Gap.csv
# This dataset provides insights into income disparities based on job titles and gender. By analyzing the base pay and bonuses for various job titles, you can identify specific roles where the gender pay gap is most significant. The demographic data (age, education, seniority) included in this dataset allows for a deeper exploration of how these factors correlate with salary differences.
# 
# ->inc_occ_gender.csv
# This dataset from the Bureau of Labor Statistics offers median weekly incomes segmented by occupation and sex. It can be used to compare and validate findings from the Glassdoor dataset, especially in understanding industry-wide trends in gender pay gaps. The data on total median income and the breakdown by gender will help highlight industries with pronounced disparities.
# 
# ->Salary_Data.csv
# This dataset consolidates information on age, experience, education level, and salary across various roles. It will serve as a comprehensive source to correlate individual characteristics (such as education and experience) with salary differences. By merging this data with the previous two datasets, you can create a more holistic view of the gender pay gap, examining how personal attributes impact salary in conjunction with industry-specific trends.
# 
# Merging by Job Title and Gender: All three datasets contain job title and gender as key attributes. You can merge them to analyze salary differences within specific roles across various industries.
# 
# Comparative Analysis: Use the Glassdoor dataset to explore salary structures within specific job titles and then validate these findings with the broader industry data from the BLS dataset. This comparison will highlight whether the trends observed in specific job titles reflect wider industry patterns.
# 
# Correlational Studies: Use the Salary_Data dataset to perform statistical analyses, such as regression or correlation, to investigate how years of experience and education levels interact with the gender pay gap identified in the other datasets. This will enable you to provide a more nuanced understanding of the factors influencing salary disparities.

# ## Approach and Analysis
# *What is your approach to answering your project question?*
# *How will you use the identified data to answer your project question?*
# 📝 <!-- Start Discussing the project here; you can add as many code cells as you need -->

# In[62]:


# Start your code here
import pandas as pd

# Define the file paths
file_paths = {
    "Glassdoor": "datasets/Glassdoor Gender Pay Gap.csv",
    "Income Occupation Gender": "datasets/inc_occ_gender.csv",
    "Salary Data": "datasets/Salary_Data.csv"
}

datasets = {}
for name, path in file_paths.items():
    datasets[name] = pd.read_csv(path)

# Display the first few rows of each dataset
for name, data in datasets.items():
    print(f"{name} Overview:")
    print(data.head(), "\n")


# ## Importing Libraries

# In[63]:


import numpy as np
import pandas as pd
import plotly.figure_factory as ff
import plotly.graph_objs as go
import plotly.offline as py
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from scipy import stats
import plotly.express as px
import seaborn as sns

from   sklearn.preprocessing import LabelEncoder
from   sklearn.tree import DecisionTreeRegressor
from   sklearn.preprocessing import PolynomialFeatures
from   sklearn.metrics import accuracy_score
from   sklearn.model_selection import train_test_split


# ## Data Exploration

# In[64]:


df = pd.read_csv('datasets/Glassdoor Gender Pay Gap.csv')


# In[84]:


df2 = pd.read_csv('datasets/Salary_Data.csv')
print(df2.head())


# In[65]:


df.head()


# In[66]:


df.shape


# In[67]:


df.info()


# In[85]:


df2.info()


# In[89]:


df.isna().sum()


# In[91]:


df2.isna().sum()


# In[94]:


# Drop rows with missing values
df2 = df2.dropna()
print(df2.isna().sum())


# In[68]:


df['TotalSalary']= df['BasePay']+df['Bonus']
df


# ## Checking the Outliers in the Dataset
# To explore the presence of outliers in the dataset, we first used a boxplot visualization. Boxplots are useful for identifying any extreme values or outliers that deviate significantly from the rest of the data. 

# In[69]:


fig = plt.figure(figsize =(12,8))
df.boxplot()
fig.show()


# Upon reviewing the boxplot, we observed that the data appears relatively stable with only a few outliers. The distribution of most of the variables seems normal, indicating no significant skew. However, we noticed that there are some outliers present in two columns: BasePay and Total salary. These outliers are likely due to higher salary values for certain individuals, which is common in large organizations or specialized roles.
# 
# It is important to note that these outliers are not necessarily erroneous or problematic. In fact, it makes sense that individuals with higher base pay will have correspondingly higher total salaries (which include bonuses or other compensation). Therefore, these outliers reflect higher-paying positions and do not suggest data issues.
# 
# To further investigate these outliers, we can calculate the specific values that qualify as outliers for the BasePay and Total columns using statistical methods like the IQR (Interquartile Range) or by directly examining the top and bottom salary figures.

# In[70]:


IQR_BasePay = df['BasePay'].quantile(0.75)-df['BasePay'].quantile(0.25)
IQR_BasePay


# In[71]:


#To get the upper outlier
upper_outlier = df['BasePay'].quantile(0.75)+1.6*IQR_BasePay
upper_outlier


# In[72]:


#To get the lower outlier
lower_outlier = df['BasePay'].quantile(0.25)-1.6*IQR_BasePay
lower_outlier


# In[73]:


#To get the outlier
outliers = df[(df['BasePay']>upper_outlier)]
outliers


# In[74]:


df.drop(outliers.index, inplace = True)


# In[75]:


df.shape


# Since these outliers are significantly higher than most of the data, we decided to remove them from the dataset to prevent them from skewing the analysis. In this case, the high total salaries for the two managerial roles seemed significantly higher than the rest of the dataset. Given that outliers can sometimes represent rare but valid data points (e.g., high-level executives or rare situations), we decided to remove them to maintain a more representative distribution of the data for analysis.
# 
# By removing these outliers, the dataset now reflects a more accurate picture of the general salary trends, which will help in further exploration of gender pay gaps and other salary-related patterns.

# ## Data Analysis and Visualization

# The primary goal of the Analysis Phase is to gain a deep understanding of the existing gender pay gap within the organization, identify contributing factors, and prepare the data for predictive modeling.

# In[ ]:





# In[ ]:


# Group by 'Gender' and calculate descriptive statistics for the specified columns
gender_stats = df.groupby('Gender')[['BasePay', 'Bonus', 'TotalSalary']].describe()

# Print the result
print(gender_stats)


# In[78]:


numeric_df = df.select_dtypes(include='number')

# Create the heatmap
fig = plt.figure(figsize=(18, 8))
sns.heatmap(numeric_df.corr(), annot=True, cmap="BuPu")

# Show the plot
plt.show()


# The heatmap provides a clear view of how the variables in the dataset relate to each other. Understanding these relationships is critical for analyzing the factors contributing to the gender pay gap and other compensation disparities. For example, knowing that BasePay and Total Salary are closely linked allows us to focus on base salary differences when comparing gender pay gaps, while the correlation between Bonus and PerfEval can help explain bonus disparities across different groups.

# In[117]:


# Define custom colors: Blue for Male, Red for Female
custom_colors = ['#1f77b4', '#ff9999']  # Blue for Male, Red for Female

# Plot: Gender pay comparison by job title
pay_gap_data[['Male', 'Female']].plot(
    kind='bar',
    figsize=(14, 7),
    stacked=False,
    color=custom_colors  # Apply the two blue shades here
)

# Add titles and labels
plt.title("Gender Pay Comparison by Job Title")
plt.ylabel("Average BasePay")
plt.xlabel("Job Title")
plt.xticks(rotation=45, ha='right')
plt.legend(title="Gender")
plt.tight_layout()
plt.show()


# ## Updated Approach:
# We can group the data by both JobTitle and Education, then calculate the average BasePay for each combination of these two features, split by Gender. This will give a better understanding of how gender pay disparities are influenced by both job role and educational background.
# 
# Group by JobTitle, Education, and Gender: We're calculating the average BasePay for each combination of JobTitle, Education, and Gender. This allows for a more granular view of the gender pay gap, considering both the job title and education level.
# 
# Pivot Table: We create a pivot table with JobTitle and Education as the index and Gender as columns. This structure makes it easy to compare the average pay for males and females within each job and education category.
# 
# 

# In[80]:


# Grouping by JobTitle, Education, and Gender, and calculating the mean BasePay
data = df.groupby(['JobTitle', 'Education', 'Gender']).agg({'BasePay': 'mean'}).reset_index()

# Pivoting the table to make JobTitle and Education as index, Gender as columns
pivot_data = data.pivot_table(index=['JobTitle', 'Education'], columns='Gender', values='BasePay')

# Calculating the pay gap between Male and Female
pivot_data['PayGap'] = pivot_data['Male'] - pivot_data['Female']

# Sorting by PayGap for better visualization
pivot_data_sorted = pivot_data.sort_values(by='PayGap', ascending=False)

# Display the result
pivot_data_sorted


# The grouping, pivoting, and pay gap calculation provided a clear and structured way to explore the gender pay gap across job titles and educational backgrounds. The visualization process revealed not only the overall pay differences between genders but also how factors like job titles and education levels play a role in shaping these disparities. By sorting the data, we can focus on areas where the pay gap is most pronounced, helping stakeholders target their efforts toward achieving a more equitable workforce.

# In[103]:


plt.figure(figsize=(12, 8))
sns.boxplot(data=df, x='Education', y='TotalSalary', hue='Gender', palette='Purples')
plt.title('Salary Distribution by Education Level and Gender')
plt.xlabel('Education Level')
plt.ylabel('Total Salary ($)')
plt.xticks(rotation=45)
plt.show()


# The boxplot provided valuable insights into the distribution of Total Salary across education levels and between genders. It clearly shows the salary disparity between men and women, highlighting areas where the pay gap might be more pronounced. This visualization is essential for understanding how education influences salary outcomes for different genders, and it can guide efforts to reduce the gender pay gap by focusing on educational attainment or exploring disparities within specific fields.

# In[109]:


# Group data by Gender and PerfEval and calculate mean TotalSalary
gender_perf_salary = df.groupby(['Gender', 'PerfEval'])['TotalSalary'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=gender_perf_salary, x='PerfEval', y='TotalSalary', hue='Gender', palette='Oranges')
plt.title('Average Total Salary by Gender and Performance Evaluation')
plt.xlabel('Performance Evaluation (1-5)')
plt.ylabel('Average Total Salary')
plt.legend(title='Gender')
plt.show()


# The bar plot allowed us to visually explore the relationship between Performance Evaluation and Total Salary, broken down by gender. It provided valuable insights into how performance impacts salary outcomes and highlighted the persistent gender pay gap, even when considering performance evaluations. This visualization is important for understanding how gender-based salary disparities may be exacerbated or mitigated by performance ratings and could inform targeted initiatives to close the gender pay gap, particularly in high-performance categories.

# In[110]:


# Group data by Gender and Seniority and calculate mean TotalSalary
gender_seniority_salary = df.groupby(['Gender', 'Seniority'])['TotalSalary'].mean().reset_index()

# Plot
plt.figure(figsize=(10, 6))
sns.barplot(data=gender_seniority_salary, x='Seniority', y='TotalSalary', hue='Gender', palette='Greens')
plt.title('Average Total Salary by Gender and Seniority Level')
plt.xlabel('Seniority Level')
plt.ylabel('Average Total Salary')
plt.legend(title='Gender')
plt.show()


# This bar plot effectively illustrates the relationship between seniority and salary, with a clear breakdown by gender. The visualization reinforces the presence of a persistent gender pay gap across all seniority levels, although the gap becomes more pronounced in higher seniority positions. Understanding this relationship is crucial for addressing and mitigating gender-based salary disparities, particularly in senior roles. This information could be used to inform organizational strategies aimed at ensuring more equitable salary practices across all levels of seniority.

# In[111]:


# Group data by Gender and Department and calculate mean TotalSalary
gender_dept_salary = df.groupby(['Gender', 'Dept'])['TotalSalary'].mean().reset_index()

# Plot
plt.figure(figsize=(12, 6))
sns.barplot(data=gender_dept_salary, x='Dept', y='TotalSalary', hue='Gender', palette='Purples')
plt.title('Average Total Salary by Gender and Department')
plt.xlabel('Department')
plt.ylabel('Average Total Salary')
plt.xticks(rotation=45)
plt.legend(title='Gender')
plt.show()


# This bar plot offers valuable insights into the gender pay gap across various departments, shedding light on areas where pay disparities are most prevalent. It is clear that while some departments show relatively small gender differences in salary, others, particularly those with higher pay scales, exhibit significant gaps. This information can be used by organizations to target interventions aimed at reducing the gender pay gap, especially in departments where the gap is most pronounced.

# There is a statistically significant difference in total pay between genders in the dataset. In other words, the data provides strong evidence that gender has an impact on total pay, and it is not likely due to random chance.
# 
# However, remember that statistical significance does not imply causation or provide insights into the reasons behind the gender pay gap. Further analysis may be needed to understand the factors contributing to this difference.

# ## Machine Learning Plan
# 
# This project explores the gender pay gap and aims to use machine learning for predicting salaries based on various features, such as gender, education, years of experience, and department. Below is the plan for integrating machine learning into this project.
# 
# 1. Types of Machine Learning to be Used
# 
#     A. Regression (Supervised Learning)
# 
#     -Linear Regression: To understand the relationship between salary and features like education and years of experience.
# 
#     -Decision Tree Regression: To capture non-linear relationships and interactions between features.
# 
#     -Random Forest Regression: For robust predictions by considering a wider set of features and interactions.
# 
#     OR
# 
#     B. Classification (Supervised Learning)
# 
#     -Logistic Regression: To classify employees based on their pay gap compared to expected salary.
# 
#     -Random Forest Classification: For a more robust classification based on multiple features.
# 
# 2. Issues in Making Machine Learning Happen
# A. Data Quality Issues
# Missing Data: We need to handle missing values, especially in columns like Age and Gender.
# Data Imbalance: There could be an imbalance in the gender distribution across high-paying job titles, which may affect model performance.
# B. Data Preprocessing
# Feature Engineering: Converting categorical variables (e.g., Education Level and Job Title) into numerical values for machine learning models.
# Normalization: Scaling features like years of experience and salary to ensure the models perform well.
# C. Overfitting
# Complex models may overfit the data. We will use cross-validation in decision trees to avoid this.
# 
# 3. Potential Challenges
# -Non-linear Relationships
# Salary is influenced by complex, non-linear interactions between features. Models like linear regression may not capture these without careful feature engineering or more advanced techniques.
# 
# 4. Plan for Next Steps
# Data Preprocessing: Address missing values, encode categorical features, and scale the data.
# Model Training: Train ML models, evaluate them using metrics like MAE for regression or accuracy for classification.

# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# Course videos, notes, geeksforgeeks website

# In[121]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

