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

# In[1]:


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


# ## Resources and References
# *What resources and references have you used for this project?*
# 📝 <!-- Answer Below -->

# In[2]:


# ⚠️ Make sure you run this cell at the end of your notebook before every submission!
get_ipython().system('jupyter nbconvert --to python source.ipynb')

