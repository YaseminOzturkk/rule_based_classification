# Calculating Potential Customer Return with Rule-Based Classification
## Rule Based Classification
Rule-based classification is an approach for categorizing data within a dataset by applying predefined rules. This method involves creating specific rules to classify the data, and then sorting the data according to these rules.

Rule-based classification is typically a straightforward and efficient technique. It operates on predetermined rules, and the classification outcomes are determined by adhering to these rules. As a result, rule-based classification can prove to be particularly useful, especially when dealing with small datasets.

## Business Problem
A gaming company wishes to create level-based new customer personas using certain characteristics of its customers and to create segments based on these new customer personas. The company aims to estimate how much the potential new customers, based on these segments, can potentially generate in terms of revenue.
For example, it is desired to determine how much a 25-year-old male user from Turkey who uses IOS can potentially generate on average.

## Dataset Story

Persona.csv data set contains the prices of the products sold by an international game company and some demographic information of the users who purchased these products. The data set consists of records created in each sales transaction. This means the table is not deduplicated. In other words, a user with certain demographic characteristics may have made more than one purchase.

- Price: Customer's spending amount
- Source: The type of device the customer is connected to
- Sex: Customer's gender
- Country: Customer's country
- Age: Customer's age
![image](https://github.com/YaseminOzturkk/rule_based_classification/assets/48058898/e3513783-0d44-4a8a-8172-b1d63670bfcb)

## Method and Libraries
- Segmentation
- Exploratory Data Analysis (EDA)
- Pandas
- Streamlit
- PIL
- Matplotlib
- Plotly

## Requirements.txt
Please examine the 'requirements.txt' file to identify the necessary libraries.

