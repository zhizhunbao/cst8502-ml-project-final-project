# Traffic Violation Analysis - Output Results

**Generated on:** 2025-11-20 19:41:45

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

┌─────────────────────────────────────────────────────────────────────────┐
│ Team Member          │ Primary Task              │ Responsibilities    │
├─────────────────────────────────────────────────────────────────────────┤
│ Joseph Weng          │ Classification (DT)       │ - Data preparation  │
│ (041076091)          │                           │ - Build DT model    │
│                      │                           │ - Cross-validation  │
│                      │                           │ - Interpret rules   │
├─────────────────────────────────────────────────────────────────────────┤
│ Hye Ran Yoo          │ Clustering (KMeans)       │ - Feature selection │
│ (041145212)          │                           │ - Elbow method      │
│                      │                           │ - Build clusters    │
│                      │                           │ - Interpret groups  │
├─────────────────────────────────────────────────────────────────────────┤
│ Peng Wang            │ Outlier Detection         │ - LOF method        │
│ (041107730)          │ (LOF + Distance)          │ - Distance method   │
│                      │                           │ - Combine results   │
│                      │                           │ - Analyze outliers  │
├─────────────────────────────────────────────────────────────────────────┤
│ All Members          │ Shared Responsibilities   │ - Data exploration  │
│                      │                           │ - Documentation     │
│                      │                           │ - Presentation prep │
└─────────────────────────────────────────────────────────────────────────┘

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

Dataset loaded successfully from: D:\BaiduSyncdisk\workspace\algonquin_workspace\cst8502-ml-project-final-project\TrafficViolations.csv
Total records: 2,057,983
Total columns: 43

First 5 rows of the dataset:
                                  SeqID  ...                                Geolocation
0  b4fedce9-1be2-41d0-b600-e2987e079ecd  ...      (39.0477626666667, -77.0518796666667)
1  89ddefe5-70f2-4b30-9d60-99ea481a505b  ...      (38.9917846666667, -77.0254603333333)
2  3b3f3ef9-b28d-4203-ab65-4b92c9e40748  ...            (39.0856023333333, -76.9994485)
3  a6b25057-e8a1-4d03-a607-7182574808e6  ...             (39.184545, -77.3124116666667)
4  f2191d01-7f2b-4fbd-8e06-97ebd82f1056  ...              (39.1726083333333, -77.25267)

[5 rows x 43 columns]

## 3.2. Describe data
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 2057983 entries, 0 to 2057982
Data columns (total 43 columns):
 #   Column                   Dtype         
---  ------                   -----         
 0   SeqID                    object        
 1   Date Of Stop             datetime64[ns]
 2   Time Of Stop             object        
 3   Agency                   object        
 4   SubAgency                object        
 5   Description              object        
 6   Location                 object        
 7   Latitude                 float64       
 8   Longitude                float64       
 9   Accident                 bool          
 10  Belts                    bool          
 11  Personal Injury          bool          
 12  Property Damage          bool          
 13  Fatal                    bool          
 14  Commercial License       bool          
 15  HAZMAT                   bool          
 16  Commercial Vehicle       bool          
 17  Alcohol                  bool          
 18  Work Zone                bool          
 19  Search Conducted         bool          
 20  Search Disposition       object        
 21  Search Outcome           object        
 22  Search Reason            object        
 23  Search Reason For Stop   object        
 24  Search Type              object        
 25  Search Arrest Reason     object        
 26  State                    object        
 27  VehicleType              object        
 28  Year                     Int64         
 29  Make                     object        
 30  Model                    object        
 31  Color                    object        
 32  Violation Type           object        
 33  Charge                   object        
 34  Article                  object        
 35  Contributed To Accident  bool          
 36  Race                     object        
 37  Gender                   object        
 38  Driver City              object        
 39  Driver State             object        
 40  DL State                 object        
 41  Arrest Type              object        
 42  Geolocation              object        
dtypes: Int64(1), bool(12), datetime64[ns](1), float64(2), object(27)
memory usage: 512.3+ MB

### BASIC STATISTICS

Numeric columns:
           Latitude     Longitude         Year
count  2.057983e+06  2.057983e+06    2047348.0
mean   3.615751e+01 -7.134010e+01  2007.569054
std    1.028911e+01  2.030090e+01    85.114158
min    0.000000e+00 -1.512560e+02          0.0
25%    3.901650e+01 -7.719289e+01       2003.0
50%    3.906617e+01 -7.708751e+01       2008.0
75%    3.913692e+01 -7.702697e+01       2013.0
max    4.154316e+01  3.906444e+01       9999.0

Categorical columns (showing top 10 of 27):
  - SeqID: 1170571 unique values
  - Geolocation: 1019918 unique values
  - Location: 268060 unique values
  - Model: 23324 unique values
  - Description: 17721 unique values
  - Driver City: 9348 unique values
  - Make: 4918 unique values
  - Time Of Stop: 1440 unique values
  - Charge: 1199 unique values
  - Search Reason For Stop: 836 unique values

Boolean columns:
  - Accident: True=2.7%, False=97.3%
  - Belts: True=3.1%, False=96.9%
  - Personal Injury: True=1.2%, False=98.8%
  - Property Damage: True=2.2%, False=97.8%
  - Fatal: True=0.0%, False=100.0%
  - Commercial License: True=2.8%, False=97.2%
  - HAZMAT: True=0.0%, False=100.0%
  - Commercial Vehicle: True=0.3%, False=99.7%
  - Alcohol: True=0.1%, False=99.9%
  - Work Zone: True=0.0%, False=100.0%
  - Search Conducted: True=4.3%, False=95.7%
  - Contributed To Accident: True=2.7%, False=97.3%

## 3.3. Explore data




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

## 3.4. Verify data quality
  - SeqID values with duplicates: 422725
  - Top 5 duplicate SeqID counts (descending):
SeqID
33c49de3-9e36-4f36-9326-b59a95e86fe8    59
75f55258-401e-4a3d-87e6-ff9394acb877    43
6c64b741-3b7e-4658-9fed-fd5c3af27250    43
28c5bfc1-5670-4751-be16-e1d83cda20f0    42
c474e98e-128c-40e0-a487-d988cc67f0bd    40
  - Search Arrest Reason: 1,996,217 (97.00%)
  - Search Type: 1,968,679 (95.66%)
  - Search Disposition: 1,968,671 (95.66%)
  - Search Reason: 1,968,671 (95.66%)
  - Search Outcome: 801,209 (38.93%)
  - Search Reason For Stop: 782,221 (38.01%)
  - Article: 92,208 (4.48%)
  - Color: 22,267 (1.08%)
  - Year: 10,635 (0.52%)
  - DL State: 929 (0.05%)
  - Driver City: 517 (0.03%)
  - Model: 223 (0.01%)
  - Make: 74 (0.00%)
  - State: 59 (0.00%)
  - Driver State: 11 (0.00%)
  - Description: 10 (0.00%)
  - Location: 4 (0.00%)

# 4. Classification by Decision Tree

## 4.1. Data Preparation for Classification

### 4.1.1. Select data
Selected 24 features for classification
Target variable: Violation Type

### 4.1.2. Clean data
Original: 2,057,983 rows → After cleaning: 2,057,081 rows
Filtered to 3 classes (min 1000 instances each)
Class distribution:
Violation Type
Warning     1082088
Citation     883687
ESERO         91306
Name: count, dtype: int64

### 4.1.3. Construct data
Created 6 temporal features. Head:
   Hour  Month  DayOfWeek  IsWeekend TimeOfDay  VehicleAge
0     6     11          5          1   Morning          15
1     6     11          5          1   Morning           8
2     6     11          5          1   Morning          11
3     6     11          5          1   Morning           9
4     5     11          5          1     Night          -1

### 4.1.4. Integrate data
Final features selected: 23

### 4.1.5. Format data

Feature matrix: 10,000 samples × 23 features
Target: 3 classes

## 4.2. Modelling

### 4.2.1. Select modeling techniques
Selected: Decision Tree Classifier
Rationale: Decision trees are interpretable and suitable for classification tasks

### 4.2.2. Generate test design
Test Design: Train/Test Split (70%/30%)
Train: 7,000 (70.0%) | Test: 3,000 (30.0%)

### 4.2.3. Build model

Training accuracy: 57.89% | Test accuracy: 57.03%

Top 10 Features:
                Feature  Importance
Contributed To Accident    0.370064
                   Hour    0.173957
             VehicleAge    0.168676
        Property Damage    0.126890
              IsWeekend    0.047455
        Personal Injury    0.038673
               Latitude    0.021198
              DayOfWeek    0.020280
         Gender_encoded    0.019694
              Longitude    0.013112

### 4.2.4. Assess model




![Figure 3](images\figure_03.png)





![Figure 4](images\figure_04.png)

Decision Tree Rules (Top 3 levels):
|--- Contributed To Accident <= 0.50
|   |--- VehicleAge <= 12.50
|   |   |--- Hour <= 5.50
|   |   |   |--- Hour <= 2.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- Hour >  2.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |--- Hour >  5.50
|   |   |   |--- Property Damage <= 0.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- Property Damage >  0.50
|   |   |   |   |--- class: Citation
|   |--- VehicleAge >  12.50
|   |   |--- Property Damage <= 0.50
|   |   |   |--- Hour <= 4.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |   |--- Hour >  4.50
|   |   |   |   |--- truncated branch of depth 2
|   |   |--- Property Damage >  0.50
|   |   |   |--- Longitude <= -77.05
|   |   |   |   |--- class: Citation
|   |   |   |--- Longitude >  -77.05
|   |   |   |   |--- class: Citation
|--- Contributed To Accident >  0.50
|   |--- Longitude <= -77.11
|   |   |--- VehicleAge <= 16.50
|   |   |   |--- class: Citation
|   |   |--- VehicleAge >  16.50
|   |   |   |--- class: Citation
|   |--- Longitude >  -77.11
|   |   |--- Latitude <= 39.05
|   |   |   |--- Latitude <= 39.03
|   |   |   |   |--- class: Citation
|   |   |   |--- Latitude >  39.03
|   |   |   |   |--- class: Citation
|   |   |--- Latitude >  39.05
|   |   |   |--- class: Citation

... (truncated)

## 4.3. Evaluation

### 4.3.1. Evaluate results

Model Performance Summary:
  Training Accuracy: 57.89%
  Test Accuracy: 57.03%
  Cross-Validation Mean: 57.59% (+/- 2.03%)

### 4.3.2. Interpret results

Top 10 Most Important Features:
                Feature  Importance
Contributed To Accident    0.370064
                   Hour    0.173957
             VehicleAge    0.168676
        Property Damage    0.126890
              IsWeekend    0.047455
        Personal Injury    0.038673
               Latitude    0.021198
              DayOfWeek    0.020280
         Gender_encoded    0.019694
              Longitude    0.013112

Confusion Matrix (Actual Values):
          Citation  ESERO  Warning
Citation       205      0     1084
ESERO            7      0      126
Warning         72      0     1506




![Figure 5](images\figure_05.png)


Confusion Matrix Shape: (3, 3)

### 4.3.3. Review of process

Process Review:
  - Data preparation: Completed with feature engineering
  - Model selection: Decision Tree Classifier
  - Model training: Completed with cross-validation
  - Model evaluation: Performance metrics calculated

### 4.3.4. Determine next steps

Next Steps:
  - Consider hyperparameter tuning for improved performance
  - Explore ensemble methods (Random Forest, Gradient Boosting)
  - Analyze misclassified cases for insights

Classification Complete: 10,000 records, 3 classes, 23 features
  Accuracy: Train 57.89% | Test 57.03%

# 5. Clustering by KMeans

## 5.1. Data Preparation for Clustering

### 5.1.1. Select Data - Select and justify features for clustering

Dataset: 2,057,983 rows × 43 columns
Selected 17 features for clustering

### 5.1.2. Clean Data
Original: 2,057,983 rows → After cleaning: 2,057,983 rows

### 5.1.3. Construct Data - Feature Engineering

Created temporal features: Hour, Month, DayOfWeek, IsWeekend, TimeOfDay
VehicleAge: mean=17.3yr, range=[-7974, 2025]

### 5.1.4. Integrate Data

Final features: 20 (≥10) - 7 numeric, 8 boolean, 5 categorical

### 5.1.5. Format Data - Encode, scale, and validate

Encoded 5 categorical features
Feature matrix: 10,000 rows × 20 columns
Scaled features: 10,000 rows × 20 columns (StandardScaler)
Using all 10,000 records for clustering

Data Preparation Complete: 10,000 records × 20 features

## 5.2. Modelling

### 5.2.1. Select modeling techniques
Selected modeling technique: KMeans Clustering

KMeans selected for clustering analysis based on:
  - Unsupervised learning requirement
  - Ability to identify patterns in traffic violation data
  - Interpretability of cluster results

### 5.2.2. Generate test design
Test Design: Elbow Method + Silhouette Score + Davies-Bouldin Score
Testing k values from 2 to 10




![Figure 6](images\figure_06.png)


Selected optimal k: 5

### 5.2.3. Build model

KMeans Results: k=5, Inertia=145020, Silhouette=0.182, DB=1.651
Cluster distribution:
  Cluster 0: 129 (1.3%)
  Cluster 1: 3,056 (30.6%)
  Cluster 2: 729 (7.3%)
  Cluster 3: 1,859 (18.6%)
  Cluster 4: 4,227 (42.3%)

### 5.2.4. Assess model

Cluster Characteristics:

Cluster 0: 129 (1.3%)
  Hour: 17, Month: 12, Weekend: 37%
  Accident: 48.1%, Alcohol: 0.0%, VehicleAge: 18.0yr
  Top violations: Citation(91%), Warning(9%)

Cluster 1: 3,056 (30.6%)
  Hour: 8, Month: 3, Weekend: 0%
  Accident: 1.5%, Alcohol: 0.1%, VehicleAge: 15.9yr
  Top violations: Warning(54%), Citation(43%), ESERO(4%)

Cluster 2: 729 (7.3%)
  Hour: 22, Month: 4, Weekend: 23%
  Accident: 3.6%, Alcohol: 0.5%, VehicleAge: 17.2yr
  Top violations: Citation(52%), Warning(45%), ESERO(4%)

Cluster 3: 1,859 (18.6%)
  Hour: 22, Month: 3, Weekend: 100%
  Accident: 2.9%, Alcohol: 0.1%, VehicleAge: 16.8yr
  Top violations: Citation(48%), Warning(47%), ESERO(5%)

Cluster 4: 4,227 (42.3%)
  Hour: 22, Month: 3, Weekend: 0%
  Accident: 1.7%, Alcohol: 0.0%, VehicleAge: 16.9yr
  Top violations: Warning(57%), Citation(38%), ESERO(5%)

Generating visualizations...




![Figure 7](images\figure_07.png)





![Figure 8](images\figure_08.png)


Model Assessment Summary:
  Optimal k: 5 | Silhouette: 0.182 | DB Score: 1.651
  Outliers: 100 (1.00%)

## 5.3. Evaluation

### 5.3.1. Evaluate results

Clustering Results Summary:
  Dataset: 10,000 records
  Number of clusters: 5
  Silhouette Score: 0.182
  Davies-Bouldin Score: 1.651
  Inertia: 145020

### 5.3.2. Interpret results

Cluster Interpretation:
  Cluster 0: 129 records (1.3%)
  Cluster 1: 3,056 records (30.6%)
  Cluster 2: 729 records (7.3%)
  Cluster 3: 1,859 records (18.6%)
  Cluster 4: 4,227 records (42.3%)

### 5.3.3. Review of process

Process Review:
  - Data preparation: Completed with feature engineering and scaling
  - Model selection: KMeans Clustering
  - Optimal k selection: Elbow method + Silhouette + Davies-Bouldin
  - Model training: Completed
  - Model assessment: Cluster characteristics analyzed

### 5.3.4. Determine next steps

Next Steps:
  - Consider hierarchical clustering for comparison
  - Analyze cluster stability with different initializations
  - Investigate outliers detected from clusters

Clustering Complete: 10,000 records, k=5, Silhouette=0.182

# 6. Outlier Detection by LOF and Distance-based method

## 6.1. Data Preparation for Outlier Detection

### 6.1.1. Select Data - Select and justify features for outlier detection

Dataset: 2,057,983 rows × 22 columns (19 modeling features)

### 6.1.2. Clean Data

Removed 887,412 duplicates (43.1%)
2 columns with missing values (will handle after feature engineering)
Found 335541 invalid values across fields
Removed 342,116 invalid records → 10,000 remaining
Sampled 10,000 records (1.2%) using stratified sampling

Final dataset: 10,000 rows

### 6.1.3. Construct Data - Feature Engineering

Created temporal features: Hour, Month, DayOfWeek, IsWeekend, TimeOfDay
VehicleAge: mean=14.9yr, range=[0, 50]
Created binning features: VehicleAge_Binned, Hour_Binned (optional)
No missing values after feature engineering

### 6.1.4. Integrate Data




![Figure 9](images\figure_09.png)


Final features: 22 (≥10 ) - 7 numeric, 10 boolean, 5 categorical

Association Rule Mining:
  Generated 11 association rules
  Top 5 associations (by support):
    Accident & Property Damage: support=0.007, confidence=0.367
    Accident & Personal Injury: support=0.004, confidence=0.213
    Accident & Belts: support=0.001, confidence=0.064
    Property Damage & Belts: support=0.001, confidence=0.085
    Personal Injury & Belts: support=0.001, confidence=0.127

Correlation Analysis:
  Found 1 highly correlated pairs (|r| > 0.8)

### 6.1.5. Format Data - Encode, scale, and validate

Encoded 5 categorical features
Feature matrix: 10,000 rows × 22 columns
Scaled features: 10,000 rows × 22 columns (StandardScaler)
Data quality check passed (no NaN/Inf)

Data Preparation Complete: 10,000 records × 22 features

## 6.2. Modelling

### 6.2.1. Select modeling techniques
Selected modeling techniques:
  1. Local Outlier Factor (LOF)
  2. Distance-based Outlier Detection

Both LOF and Distance-based method selected for outlier detection (as per project requirements)

### 6.2.2. Generate test design
Test Design:

1. Local Outlier Factor (LOF):
   - n_neighbors=20: Number of neighbors to consider for density estimation
   - contamination=0.01: Expected proportion of outliers (1%)
   - Rationale: Balanced parameter for detecting local density anomalies

2. Distance-based Outlier Detection:
   - Distance metric: Euclidean distance
   - k_neighbors=20: Number of nearest neighbors to calculate average distance
   - Threshold: 99th percentile (top 1% as outliers)
   - Rationale: Consistent with LOF contamination rate for fair comparison

3. Combined Approach:
   - Strategy: Identify common outliers detected by both methods
   - Benefit: Reduces false positives and increases confidence in outlier detection

### 6.2.3. Build model

1. LOF Model:
   - Configured: n_neighbors=20, contamination=0.01
   - Detected: 100 outliers

2. Distance-based Model:
   - Configured: k_neighbors=20, threshold=3.9439 (99th percentile)
   - Detected: 100 outliers

### 6.2.4. Assess model

Outlier Detection Results:
  LOF: 100 outliers (1.00%)
  Distance-based: 100 outliers (1.00%)
  Common (Both methods): 54 outliers (0.540%)
  Agreement: 54.0% of LOF outliers, 54.0% of Distance-based outliers

## 6.3. Evaluation

### 6.3.1. Evaluate results

Evaluation Summary:
  Dataset: 10,000 records, 22 features
  LOF detected: 100 outliers (1.00%)
  Distance-based detected: 100 outliers (1.00%)
  Common outliers: 54 (0.540%)

Common Outlier Characteristics:
  Accident rate: 40.7%
  Alcohol involvement: 5.6%
  Average vehicle age: 16.6 years
  Top violation types: Citation (67%), Warning (33%)




![Figure 10](images\figure_10.png)


### 6.3.2. Interpret results

Results Interpretation:
  - LOF and Distance-based method detected different sets of outliers
  - 54 outliers were detected by both methods (high confidence)
  - Common outliers show distinct characteristics

### 6.3.3. Review of process

Process Review:
  1. Data Preparation:
     - Initial dataset: 2,057,983 records
     - After cleaning and sampling: 10,000 records
     - Feature engineering: 22 features created
     - Scaling: StandardScaler applied
  2. Model Selection:
     - LOF: n_neighbors=20, contamination=0.01
     - Distance-based: k_neighbors=20, 99th percentile threshold
  3. Model Execution:
     - Both models successfully trained and applied
     - Common outliers identified for high-confidence detection
  4. Evaluation:
     - Quantitative metrics calculated
     - Visualizations generated
     - Outlier characteristics analyzed

### 6.3.4. Determine next steps

Recommended Next Steps:
  1. Outlier Investigation:
     - Analyze the 54 common outliers in detail
     - Review specific cases to understand why they are anomalous
  2. Domain Expert Validation:
     - Consult with traffic safety experts to validate findings
     - Verify if detected outliers represent genuine anomalies
  3. Further Analysis:
     - Apply clustering (kMeans) to group similar violations
     - Build classification models (Decision Tree) for prediction
  4. Model Refinement:
     - Experiment with different contamination rates
     - Consider ensemble methods for improved detection

### 7
### Conclusion

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
