# CST8502 FinalProject 25F

_From PDF Document_

---

## Overview

**CST 8502 Final Project (Mandatory)**

- **Part 1 due:** Nov 7
- **Part 2 due:** Nov 21
- **Presentations:** Nov 24 – Dec 5

The goal of this project is to apply the algorithms we learned this term on a real dataset. You must follow **CRISP-DM** to complete this project.

---

## Project Requirements

You must prepare your data to perform:

1. **Classification by Decision Tree (DT)**
2. **Clustering using kMeans** & Outlier detection by clustering approach
3. **Outlier detection using LOF & distances method**

### Tools

- You must do this project using **RapidMiner** or **Python**
- Any other tool or language is **not acceptable**
- As a team, you must decide either RM or Python, but not both
- If all team members are not ready to use Python, then it is advisable to use RapidMiner

---

## Dataset List

| Dataset                           | Groups     |
| --------------------------------- | ---------- |
| Dallas Police Incidents           | 1, 2, 3, 4 |
| Dallas 911 Calls Burglary         | 5, 6, 7    |
| Austin Crash Report Data          | 8, 9, 10   |
| Montgomery Traffic violations     | 11, 12, 13 |
| Austin Crime Reports 2018         | 14         |
| New York Motor Vehicle Collisions | 15, 16     |

Groups are published in Brightspace. You can see your group number in the announcement and Content/Presentation Schedule.

---

## Submission Requirements

- The report should be typed in **OneDrive Word Doc** and shared with: thomasa@algonquincollege.com
- Instructor should be able to see the version-history
- **First Step:** Create a OneDrive Word Doc and share with: thomasa@algonquincollege.com
- **DO NOT Zip files** - zipped files will not be graded

---

## Part 1 – Business Understanding, Data Understanding & Data Preparation

### Overview

This project should be done in 2 parts. As Part 1, you will be working on:

- Business Understanding
- Data Understanding
- Data Preparation

### Deliverables

You need to work on the given dataset (check the dataset table to find your dataset based on your project group number) and propose a data science project that can be done for the given data.

### Tasks

1. **Classification Task:** Frame a question that you want to answer by your analysis

   - This question should **not** be something that can be easily answered using Excel

2. **Main Tasks:**

   - Classification by DT
   - Clustering
   - Outlier detection

3. **Project Plan:** Must explain how you are going to complete the project
   - Each task should be done by one student
   - Detailed workload distribution table should be included

### Data Understanding & Preparation

#### Association and Correlation Analysis

Use **Association Rule Mining** and **Correlation Matrix Analysis** to identify relationships between attributes. These methods help to uncover hidden patterns and dependencies within the dataset, aiding in better feature selection and model performance.

#### General Preparation Steps

1. Removing duplicates
2. Generating or assigning ID
3. Setting the correct data types for the attributes based on meaning
4. Setting the role for class attribute
5. etc.

#### Model-Specific Preparations

- Selection of attributes that can contribute to your task
  - Attributes good for clustering may not be good for outlier detection
- Binning
- Scaling
- Type conversions
- Handling missing data
- etc.

#### Special Instructions

- If you have latitude and longitude columns, make sure to do a clustering only for those columns to create regions out of it
- When you create bins, make sure to create **less than 10 bins** – 5 to 8 should be good enough in most cases

### Data Filtering & Sampling

- If you have a lot of data from multiple years, consider data from the latest year
  - Don't filter for 2025 as we don't have the data for the full year
- Even after filtering, if you have a lot of data, apply **sampling techniques (stratified)** to get a sample of **10,000 instances**

### Attributes Requirements

- **At least 10 relevant attributes** (minimum requirement)
- More relevant attributes lead to meaningful results
- You can create new attributes
  - Example: If you have a date-time column, you can create:
    - Date
    - Time
    - Month
    - Time of day (morning, afternoon, evening, night, late night, etc.)
    - Weekday/weekend
    - Day
    - etc.
- Make sure that you **don't have redundant data**
- More attributes will give you better results

### Project Title

- You must choose a title for your project that reflects its goal
- Be creative and ensure the title effectively conveys the purpose of your analysis

### Report Format

- The document should have a **cover page** (should include student names and numbers, project title, etc.)
- Follow CRISP-DM when you do this task
- **Every step of each phase** must be documented in the project report
- **Professional report style:**
  - Font: Times New Roman
  - Size: 12
  - Line spacing: 1.5
  - Justified

### Sample Project

For example, if we have a crime dataset that has information about:

- Victim (age, sex, race)
- Offender (age, sex, race)
- Time, location
- Whether the person died or not
- Number of people involved
- Number of officers involved
- Weapons involved
- etc.

**Question:** "What are the contribution factors for a crime to end up in fatality?"

**Analysis:**

- Create a decision tree with these factors by considering the fatality column as the class
- Detect outliers (e.g., victim is a child) using outlier detection methods
- Cluster instances using clustering techniques
- Similar crimes based on type, location, time, season, etc. will be grouped together

---

## Part 2 – Modeling & Evaluation

### Tasks

Apply different modeling techniques for:

1. **Outlier Detection:** LOF, distances

   - If outlier detection by distance takes too long, choose any other outlier detection approach available in RM
   - You must use 2 approaches and combine results to get common outliers

2. **Clustering:** kMeans

   - Cluster and find outliers from clusters
   - Use elbow method to find best k

3. **Classification:** DT
   - Use cross-validation for DT classification

### Documentation Requirements

- All steps must be reported in a professional style
- When building prediction models, provide detailed screenshots of results
- Describe accuracy by presenting:
  - Confusion matrices
  - R2 values
  - etc.

### Interpretation Requirements

Provide interpretation for:

- **Classification results:** Rules of DT
- **Clustering results:** Why instances are clustered together, any patterns in clusters
- **Outlier detection results:** Reason why instances are outliers

**Minimum:** Interpret at least a few clusters and at least a few outlier instances

---

## Project Presentation

### Schedule

During the last 2 weeks of the term, you will present your final project.

- Schedule: Available under Content ➞ Presentation Schedule

### Presentation Requirements

- **Duration:** 30-minute presentation
- **Tool:** PowerPoint slides
- **Per student:** ~10 minutes
  - Team with 2 members: 20 minutes
  - Solo project: 10 minutes
  - **Penalty:** If a student uses more than 10 minutes

### Content

Briefly describe:

1. Your dataset
2. The question answered by your analysis
3. Various data understanding and preparation steps
4. Your analysis and results (mention algorithms used)
5. Whether analysis confirmed or denied expectations
6. Any surprises found
7. Analysis of accuracy and importance
8. Interpretation of results (DT rules, clustering patterns, outlier reasons)

### Required Sections

- Introduction
- Business Understanding
- Data Understanding
- Data Preparation
- Modeling
- Discussion of Results
- Conclusion

---

## General Expectations

### Individual Contributions

- Each student's marks will be based primarily on their **individual contributions**
- Even though this is a group project, every student must:
  - Independently complete all sub-steps of Data Understanding phase for **one third of the columns**
  - Perform all required steps in Data Preparation phase based on their chosen model
  - Perform their own modeling, tuning parameters to achieve optimal performance
  - Validate and evaluate their model & results

### Attribute Selection

- Attributes selected for clustering may not be suitable for outlier detection and vice versa
- Choose your attributes based on your model

### Documentation

Students must document their entire process, including:

- All steps
- Assumptions
- Approaches
- Challenges
- Solutions
- Results

Each student is responsible for writing their individual contributions in the report.

### Submission Requirements

**Final submission must include:**

1. Presentation PPT
2. Consolidated RMP file (or py file if entire team chooses Python)
3. Final report

### Evaluation

- Every student will be evaluated for only one task from the given tasks:
  - Classification
  - Clustering
  - Outlier detection

In the report, make sure to include screenshots of corresponding subprocesses in corresponding sections.

### Process Template

The template for the entire process MUST look as follows (if using python, make sure to follow the same approach).

---

## Submission

### Part 1 (Nov 7th, 2025)

Submit:

- RMP or Python files
- Report (Sections 1-3, 4.1, 5.1, and 6.1)

**DO NOT Zip files**

### Part 2 (Nov 21st, 2025)

Submit as continuation of Part 1 files:

- RMP or Python files
- Report (remaining sections)
- PPT files

**DO NOT zip files**

### Presentation

Week 11 & 12 (Nov 24 - Dec 5)

- Check Brightspace for presentation schedule

### Important Notes

- **To get grades, BOTH submission AND presentation are required**
- Deliverables should be from the perspective of providing a report and presenting it to a company or job interview where they aren't sure what data science is about
  - Just creating some tables and pictures is not enough
- **Successful completion of the project is mandatory to pass this course**

---

## Workload Distribution

Before starting the project report, complete your workload distribution table, which is attached along with the project instructions on Brightspace. You must mention the selected tool in the Workload distribution table.

---

## Report Template Structure

The following is the recommended structure for your project report. Use this template to organize your report following CRISP-DM methodology.

---

# <<Title of the Project>>

_(title should be creative)_

**Student Names and Numbers**

**Date:** <<Date>>

---

## 1. Introduction

## 2. Business Understanding

### 2.1 Determine business objectives

### 2.2 Assess situation

### 2.3 Determine data mining goals

### 2.4 Produce project plan

## 3. Data Understanding

### 3.1 Collect initial data

### 3.2 Describe data

### 3.3 Explore data

### 3.4 Verify data quality

## 4. Classification by Decision Trees

### 4.1 Data Preparation

#### 4.1.1 Select data

#### 4.1.2 Clean data

#### 4.1.3 Construct data

#### 4.1.4 Integrate data

#### 4.1.5 Format data

### 4.2 Modelling

#### 4.2.1 Select modeling techniques

#### 4.2.2 Generate test design

#### 4.2.3 Build model

#### 4.2.4 Assess model

### 4.3 Evaluation

#### 4.3.1 Evaluate results

#### 4.3.2 Interpret results

#### 4.3.3 Review of process

#### 4.3.4 Determine next steps

## 5. Clustering by kMeans (clustering and finding outliers)

### 5.1 Data Preparation

#### 5.1.1 Select data

#### 5.1.2 Clean data

#### 5.1.3 Construct data

#### 5.1.4 Integrate data

#### 5.1.5 Format data

### 5.2 Modelling

#### 5.2.1 Select modeling techniques

#### 5.2.2 Generate test design

#### 5.2.3 Build model

#### 5.2.4 Assess model

### 5.3 Evaluation

#### 5.3.1 Evaluate results

#### 5.3.2 Interpret results

#### 5.3.3 Review of process

#### 5.3.4 Determine next steps

## 6. Outlier Detection by LOF and ISF (and common outliers)

### 6.1 Data Preparation

#### 6.1.1 Select data

#### 6.1.2 Clean data

#### 6.1.3 Construct data

#### 6.1.4 Integrate data

#### 6.1.5 Format data

### 6.2 Modelling

#### 6.2.1 Select modeling techniques

#### 6.2.2 Generate test design

#### 6.2.3 Build model

#### 6.2.4 Assess model

### 6.3 Evaluation

#### 6.3.1 Evaluate results

#### 6.3.2 Interpret results

#### 6.3.3 Review of process

#### 6.3.4 Determine next steps

## 7. Conclusion

---

**Note:** You must update your table of contents to reflect correct page numbers.