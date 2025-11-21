# Traffic Violation Analysis - Output Results

**Generated on:** 2025-11-20 20:40:25

---


# 1. Introduction

Traffic violations present significant challenges for urban traffic management
systems worldwide. Understanding patterns and contributing factors in traffic
violations can help law enforcement agencies allocate resources more effectively,
improve road safety, and reduce traffic-related incidents.

This project applies machine learning techniques to analyze traffic violation
data from Montgomery County, aiming to identify key factors that contribute to
different types of violations and detect anomalous patterns in the data.

Project Title: Discovering Contributing Factors to Traffic Violations -
A Comprehensive Machine Learning Approach Using Classification, Clustering,
and Anomaly Detection

Authors:
- Joseph Weng - 041076091
- Hye Ran Yoo - 041145212
- Peng Wang - 041107730


# 2. Business Understanding

## 2.1. Determine business objectives

Classification Question:
"What are the key contributing factors that lead to different types of traffic
violations in Montgomery County?"

This question will be answered through classification analysis using decision
trees, which will identify the most important factors (such as time, location,
vehicle characteristics, etc.) that determine different violation types.

Primary Business Objectives:

1. Enhance Public Safety
   - Identify patterns in traffic violations to improve road safety
   - Reduce traffic-related accidents and incidents

2. Optimize Resource Allocation
   - Help law enforcement agencies allocate patrol resources more efficiently
   - Focus enforcement efforts on high-risk areas and time periods

3. Improve Traffic Management
   - Understand contributing factors to traffic violations
   - Develop targeted prevention strategies based on data insights

4. Detect Anomalies
   - Identify unusual patterns or outliers that may indicate:
     * Fraudulent activities
     * Data quality issues
     * Systemic problems requiring investigation


## 2.2. Assess situation

Available Resources:
- Montgomery County Traffic Violations dataset (covering multiple years)
- Machine learning expertise and tools (Python, scikit-learn, pandas, etc.)
- Computing resources for data processing and analysis
- Team of 3 data scientists with complementary skills

Constraints:
- Data quality and completeness may vary across different time periods
- Privacy considerations in handling personal and location information
- Need to balance between data granularity and computational efficiency
- Project timeline: Part 1 due Nov 7, Part 2 due Nov 21, Presentations Nov 24-Dec 5

Assumptions:
- Historical traffic violation patterns can inform future prevention strategies
- Data contains sufficient information to identify meaningful patterns
- Violations are consistently reported and recorded across the dataset timeframe
- Selected features have predictive power for violation types

Risks:
- Data availability risks: Data may be incomplete, inconsistent across time
  periods, or have missing critical fields that could affect analysis quality
- Technical risks: Selected algorithms may not perform well on this dataset,
  computational resources may be insufficient for large-scale processing
- Timeline risks: Data cleaning and preparation may take longer than expected,
  model tuning may require more iterations than anticipated


## 2.3. Determine data mining goals

Required by project instructions - Three main data mining tasks:

1. Classification Task (Decision Tree)
   Goal: Build a decision tree model to identify key contributing factors that
         lead to different types of traffic violations
   Target Variable: Violation Type (class variable)
   Features: Time, location, vehicle characteristics, driver demographics, etc.
   Success Criteria:
   - Model achieves reasonable accuracy (>70%)
   - Decision tree rules are interpretable and actionable
   - Feature importance rankings provide meaningful insights

2. Clustering Task (KMeans)
   Goal: Group similar violations together to identify common patterns
   Expected Patterns:
   - High-risk time periods (rush hours, weekends, holidays)
   - Geographic hotspots (specific locations or regions)
   - Violation combinations (e.g., speeding + no seatbelt)
   Success Criteria:
   - Clustering results reveal meaningful and interpretable groups
   - Clusters show distinct characteristics
   - Optimal k value determined using elbow method

3. Outlier Detection (LOF + Distance-based methods)
   Goal: Identify anomalous violations that deviate significantly from normal
         patterns
   Potential Outliers:
   - Data errors or inconsistencies
   - Unusual circumstances (extreme weather, special events)
   - Fraudulent activities requiring investigation
   Success Criteria:
   - Outlier detection identifies genuinely unusual cases
   - Both methods (LOF and distance-based) produce consistent results
   - Detected outliers warrant further investigation

Data Mining Success Criteria:
- All three tasks completed with documented methodology
- Results are interpretable and provide actionable insights
- Models validated using appropriate evaluation metrics
- Cross-validation applied for classification task


## 2.4. Produce project plan

Project Implementation Plan:

Tool Selection: Python (scikit-learn, pandas, numpy, matplotlib, seaborn)

Workload Distribution:

| Team Member | Primary Task | Responsibilities |
|-------------|--------------|------------------|
| **Joseph Weng** (041076091) | Classification (Decision Tree) | • Data preparation<br>• Build DT model<br>• Cross-validation<br>• Interpret rules |
| **Hye Ran Yoo** (041145212) | Clustering (KMeans) | • Feature selection<br>• Elbow method<br>• Build clusters<br>• Interpret groups |
| **Peng Wang** (041107730) | Outlier Detection (LOF + Distance) | • LOF method<br>• Distance method<br>• Combine results<br>• Analyze outliers |
| **All Members** | Shared Responsibilities | • Data exploration<br>• Documentation<br>• Presentation prep |

Project Timeline:

Phase 1: Business Understanding & Data Understanding (Week 1-2)
- Define business objectives and data mining goals
- Collect and explore initial data
- Perform data quality assessment
- Due: November 7, 2025

Phase 2: Data Preparation (Week 2-3)
- Clean and preprocess data
- Handle missing values and outliers
- Feature engineering and selection
- Create separate datasets for each task

Phase 3: Modeling & Evaluation (Week 3-4)
- Build classification model (Joseph)
- Build clustering model (Hye Ran)
- Build outlier detection models (Peng)
- Validate and tune models
- Due: November 21, 2025

Phase 4: Presentation Preparation (Week 4-5)
- Interpret results and extract insights
- Create presentation slides
- Practice presentation
- Presentations: November 24 - December 5, 2025

Key Milestones:
✓ Part 1 Submission: November 7, 2025
  - Sections 1-3 (Introduction, Business Understanding, Data Understanding)
  - Sections 4.1, 5.1, 6.1 (Data Preparation for each task)

✓ Part 2 Submission: November 21, 2025
  - Complete modeling and evaluation sections
  - Final report and code files

✓ Final Presentation: November 24 - December 5, 2025
  - 30-minute team presentation
  - ~10 minutes per team member


# 3. Data Understanding

## 3.1. Collect initial data

Dataset loaded: 2,057,983 records × 43 columns from D:\BaiduSyncdisk\workspace\algonquin_workspace\cst8502-ml-project-final-project\TrafficViolations.csv

## 3.2. Describe data

Data description: 3 numeric, 27 categorical, 12 boolean columns

## 3.3. Explore data

Data exploration: 3 numeric, 12 boolean, 26 categorical variables visualized




![Figure 1](images\figure_01.png)





![Figure 2](images\figure_02.png)


Time Of Stop:
  - Unique values: 1,440
  - Missing values: 0 (0.00%)
  - Most frequent:
    * 23:30:00: 2,996 (0.1%)
  - Note: High cardinality (1,440 unique values) - consider grouping for analysis

Agency:
  - Unique values: 1
  - Missing values: 0 (0.00%)
  - Top values:
    * MCP: 2,057,983 (100.0%)

SubAgency:
  - Unique values: 9
  - Missing values: 0 (0.00%)
  - Top values:
    * 4th District, Wheaton: 447,698 (21.8%)
    * 3rd District, Silver Spring: 372,089 (18.1%)
    * 2nd District, Bethesda: 323,290 (15.7%)
    * 6th District, Gaithersburg / Montgomery Village: 260,216 (12.6%)
    * 5th District, Germantown: 244,108 (11.9%)
    * 1st District, Rockville: 239,625 (11.6%)
    * Headquarters and Special Operations: 170,947 (8.3%)
    * W15: 7 (0.0%)
    * S15: 3 (0.0%)

Description:
  - Unique values: 17,721
  - Missing values: 10 (0.00%)
  - Most frequent:
    * DRIVER FAILURE TO OBEY PROPERLY PLACED TRAFFIC CONTROL DEVICE INSTRUCTIONS: 170,319 (8.3%)
  - Note: High cardinality (17,721 unique values) - consider grouping for analysis

Location:
  - Unique values: 268,060
  - Missing values: 4 (0.00%)
  - Most frequent:
    * MONTGOMERY VILLAGE AVE @ RUSSELL AVE: 2,447 (0.1%)
  - Note: High cardinality (268,060 unique values) - consider grouping for analysis

Search Disposition:
  - Unique values: 7
  - Missing values: 1,968,671 (95.66%)
  - Top values:
    * Nothing: 38,752 (1.9%)
    * Contraband Only: 22,814 (1.1%)
    * Property Only: 14,763 (0.7%)
    * Contraband and Property: 12,964 (0.6%)
    * DUI: 12 (0.0%)
    * marijuana: 4 (0.0%)
    * nothing: 3 (0.0%)

Search Outcome:
  - Unique values: 5
  - Missing values: 801,209 (38.93%)
  - Top values:
    * Warning: 633,691 (30.8%)
    * Citation: 523,988 (25.5%)
    * Arrest: 62,634 (3.0%)
    * SERO: 36,458 (1.8%)
    * Recovered Evidence: 3 (0.0%)

Search Reason:
  - Unique values: 10
  - Missing values: 1,968,671 (95.66%)
  - Top values:
    * Incident to Arrest: 51,592 (2.5%)
    * Probable Cause: 21,676 (1.1%)
    * Consensual: 12,409 (0.6%)
    * K-9: 1,969 (0.1%)
    * Other: 1,119 (0.1%)
    * Exigent Circumstances: 535 (0.0%)
    * Probable Cause for CDS: 5 (0.0%)
    * plain view marijuana: 3 (0.0%)
    * Arrest/Tow: 3 (0.0%)
    * DUI: 1 (0.0%)

Search Reason For Stop:
  - Unique values: 836
  - Missing values: 782,221 (38.01%)
  - Most frequent:
    * 21-201(a1): 153,138 (7.4%)
  - Note: High cardinality (836 unique values) - consider grouping for analysis

Search Type:
  - Unique values: 6
  - Missing values: 1,968,679 (95.66%)
  - Top values:
    * Both: 66,906 (3.3%)
    * Property: 11,952 (0.6%)
    * Person: 10,436 (0.5%)
    * car: 4 (0.0%)
    * PC: 3 (0.0%)
    * Search Incidental: 3 (0.0%)

Data quality: 887,412 duplicates, 17 columns with missing values




![Figure 3](images\figure_03.png)


# 4. Classification by Decision Tree

## 4.1. Data Preparation for Classification

### 4.1.1. Select data

Classification setup: 24 features selected for predicting Violation Type




![Figure 4](images\figure_04.png)


### 4.1.2. Clean data

Data cleaning: 2,057,983 → 2,057,081 rows, 3 classes retained




![Figure 5](images\figure_05.png)


### 4.1.3. Construct data

Feature engineering: Created 6 new temporal and vehicle features




![Figure 6](images\figure_06.png)


### 4.1.4. Integrate data

Feature integration: 23 features integrated across 5 categories




![Figure 7](images\figure_07.png)


### 4.1.5. Format data

Data formatting: 10,000 samples × 23 features, 3 classes prepared




![Figure 8](images\figure_08.png)


## 4.2. Modelling

### 4.2.1. Select modeling techniques

Model selection: Decision Tree Classifier (interpretable, suitable for classification)

### 4.2.2. Generate test design

Test design: Train/Test split 70%/30% with stratification




![Figure 9](images\figure_09.png)


### 4.2.3. Build model

Model Performance: Training accuracy: 57.89% | Test accuracy: 57.03%




![Figure 10](images\figure_10.png)


### 4.2.4. Assess model




![Figure 11](images\figure_11.png)





![Figure 12](images\figure_12.png)


## 4.3. Evaluation

### 4.3.1. Evaluate results

Model Performance: Train 57.89% | Test 57.03% | CV 57.59% (±2.03%)




![Figure 13](images\figure_13.png)


### 4.3.2. Interpret results

Confusion Matrix: 3×3 classification results




![Figure 14](images\figure_14.png)


### 4.3.3. Review of process

Process review: Complete workflow from data preparation to model evaluation




![Figure 15](images\figure_15.png)


### 4.3.4. Determine next steps

Classification complete: 3 classes, 23 features, Test accuracy 57.03%




![Figure 16](images\figure_16.png)


# 5. Clustering by KMeans

## 5.1. Data Preparation for Clustering

### 5.1.1. Select Data - Select and justify features for clustering

Clustering setup: 17 features selected for pattern discovery




![Figure 17](images\figure_17.png)


### 5.1.2. Clean Data

Data cleaning: 2,057,983 → 2,057,983 rows (removed missing coordinates)




![Figure 18](images\figure_18.png)


### 5.1.3. Construct Data - Feature Engineering

Feature engineering: Created 6 temporal/vehicle features (mean VehicleAge=17.3yr)




![Figure 19](images\figure_19.png)


### 5.1.4. Integrate Data

Feature integration: 20 features across 5 categories




![Figure 20](images\figure_20.png)


### 5.1.5. Format Data - Encode, scale, and validate

Data formatting: Encoded 5 categorical, scaled 10,000 × 20 features




![Figure 21](images\figure_21.png)


## 5.2. Modelling

### 5.2.1. Select modeling techniques

Model selection: KMeans Clustering (unsupervised, pattern identification, interpretable)

### 5.2.2. Generate test design

Test design: Elbow Method + Silhouette + Davies-Bouldin, testing k=2-10




![Figure 22](images\figure_22.png)


Selected optimal k: 5

### 5.2.3. Build model

KMeans Clustering: k=5, Silhouette=0.182, DB=1.651, Inertia=145020




![Figure 23](images\figure_23.png)


### 5.2.4. Assess model

Cluster Characteristics: 5 clusters analyzed with key features




![Figure 24](images\figure_24.png)





![Figure 25](images\figure_25.png)





![Figure 26](images\figure_26.png)


Model Assessment: k=5, Silhouette=0.182, DB=1.651, Outliers=100 (1.00%)

## 5.3. Evaluation

### 5.3.1. Evaluate results

Clustering Results: 10,000 records, k=5, Silhouette=0.182, DB=1.651, Inertia=145020




![Figure 27](images\figure_27.png)


### 5.3.2. Interpret results

Cluster interpretation: 5 distinct patterns identified




![Figure 28](images\figure_28.png)


### 5.3.3. Review of process

Process review: Complete clustering workflow from data preparation to evaluation




![Figure 29](images\figure_29.png)


### 5.3.4. Determine next steps

Clustering complete: k=5, 10,000 records, Silhouette=0.182




![Figure 30](images\figure_30.png)


# 6. Outlier Detection by LOF and Distance-based method

## 6.1. Data Preparation for Outlier Detection

### 6.1.1. Select Data - Select and justify features for outlier detection

Outlier detection setup: 2,057,983 rows × 22 columns, 19 modeling features




![Figure 31](images\figure_31.png)


### 6.1.2. Clean Data

Data cleaning: 2,057,983 → 10,000 rows (removed missing coordinates)




![Figure 32](images\figure_32.png)


### 6.1.3. Construct Data - Feature Engineering

Feature engineering: Created temporal features (Hour, Month, DayOfWeek, IsWeekend, TimeOfDay), VehicleAge (mean=14.9yr), binning features, no missing values




![Figure 33](images\figure_33.png)


### 6.1.4. Integrate Data




![Figure 34](images\figure_34.png)


Feature integration: 22 final features (7 numeric, 10 boolean, 5 categorical), 11 association rules, 1 highly correlated pairs

### 6.1.5. Format Data - Encode, scale, and validate

Data formatting: Encoded 5 categorical, scaled 10,000 rows × 22 columns, quality check passed




![Figure 35](images\figure_35.png)


## 6.2. Modelling

### 6.2.1. Select modeling techniques

Model selection: LOF + Distance-based Outlier Detection (as per project requirements)

### 6.2.2. Generate test design

Test design: LOF (n_neighbors=20, contamination=0.01) + Distance-based (k=20, 99th percentile) → Common outliers for high confidence

### 6.2.3. Build model

Model building: LOF and Distance-based models trained on 10,000 samples




![Figure 36](images\figure_36.png)


### 6.2.4. Assess model

Outlier Detection: LOF 100 (1.00%), Distance 100 (1.00%), Common 54 (0.540%)




![Figure 37](images\figure_37.png)

Method Agreement: 54.0% of LOF outliers, 54.0% of Distance-based outliers

## 6.3. Evaluation

### 6.3.1. Evaluate results

Evaluation Summary: 10,000 records, LOF 100 (1.00%), Distance 100 (1.00%), Common 54 (0.540%)




![Figure 38](images\figure_38.png)


### 6.3.2. Interpret results

Outlier interpretation: 54 common outliers identified by both methods




![Figure 39](images\figure_39.png)


### 6.3.3. Review of process

Process review: Data preparation (10,000 records, 22 features) → Model selection → Execution → Evaluation completed




![Figure 40](images\figure_40.png)


### 6.3.4. Determine next steps

Outlier detection complete: 54 outliers, 10,000 records, 22 features




![Figure 41](images\figure_41.png)


# 7. Conclusion

This project successfully applied machine learning techniques to analyze traffic violation data
from Montgomery County, addressing the business question: 'What are the key contributing
factors that lead to different types of traffic violations?'

## Summary of Results

### Dataset Overview:
  - Initial records: 2,057,983
  - After preparation: 10,000 records
  - Features engineered: 22

### 1. Classification (Decision Tree):
   - Model: Decision Tree with Cross-Validation
   - Target Variable: Violation Type
   - Cross-Validation Accuracy: 57.59% (+/- 2.03%)
   - Key Findings:
     * Contributed To Accident is the most important feature (37.0% importance)
     * Time of day (Hour) and Vehicle Age are significant contributing factors
     * Decision tree rules provide interpretable insights for law enforcement

### 2. Clustering (KMeans):
   - Algorithm: KMeans with Elbow Method
   - Optimal clusters (k): 5
   - Silhouette Score: 0.182
   - Davies-Bouldin Score: 1.651
   - Key Findings:
     * Identified 5 distinct violation patterns
     * Geographic and temporal patterns revealed
     * Cluster outliers detected for further investigation

### 3. Outlier Detection:
   - Methods: LOF + Distance-based (combined approach)
   - LOF detected: 100 outliers (1.00%)
   - Distance-based detected: 100 outliers (1.00%)
   - High-confidence outliers (both methods): 54 (0.540%)
   - Key Findings:
     * Common outliers show distinct characteristics (40.7% accident rate)
     * Both methods agree on 54 high-confidence outliers
     * Outliers warrant further investigation for data quality and fraud detection

## Project Achievements

✓ All three required data mining tasks completed successfully
✓ Models validated using appropriate evaluation metrics
✓ Results are interpretable and provide actionable insights
✓ Cross-validation applied for classification task
✓ Association rule mining and correlation analysis performed
✓ Comprehensive documentation following CRISP-DM methodology

## Business Impact

The analysis provides valuable insights for:
  - Law enforcement resource allocation based on temporal and geographic patterns
  - Targeted prevention strategies using decision tree rules
  - Data quality improvement through outlier detection
  - Understanding contributing factors to different violation types

## Limitations and Future Work

  - Model accuracy (57%) could be improved with hyperparameter tuning
  - Consider ensemble methods (Random Forest, Gradient Boosting) for better performance
  - Expand feature engineering to include more contextual information
  - Validate outlier findings with domain experts
  - Explore additional clustering algorithms for comparison

Project Status: All required analyses completed successfully


---

## Notes

- This output was automatically generated by running the analysis script
- All visualizations are embedded at their generation points
- For complete results, refer to the generated plots and the Python script
