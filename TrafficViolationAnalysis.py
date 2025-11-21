# ============================================================================
# Import Libraries
# ============================================================================
import os
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.cluster import KMeans
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score

script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, 'TrafficViolations.csv')

# ============================================================================
# Helper Functions
# ============================================================================

# Print step header
def print_step_header(step_number, step_title, explanation=None):
    # Determine heading level based on number of dots in step_number
    dot_count = step_number.count('.')
    if dot_count == 0:
        # Level 1: # 1.
        heading_level = "#"
    elif dot_count == 1:
        # Level 2: ## 2.1.
        heading_level = "##"
    else:
        # Level 3: ### 2.1.1.
        heading_level = "###"
    
    print(f"\n{heading_level} {step_number}. {step_title}")
    if explanation:
        print(f"# Explanation: {explanation}")


# Print section title
def print_section_title(title, subtitle=None):
    # Section titles are typically level 3
    print(f"\n### {title}")
    if subtitle:
        print(f"### {subtitle}")


# Convert field to boolean type
def _convert_to_boolean(df, field_name):
    bool_map = {
        'Yes':   True,  'Y':     True,  'yes':   True,  'y':     True,
        'True':  True,  'true':  True,  'TRUE':  True,  '1':     True,
        'No':    False, 'N':     False, 'no':    False, 'n':     False,
        'False': False, 'false': False, 'FALSE': False, '0':     False
    }

    if df[field_name].dtype == 'object':
        series = df[field_name].map(bool_map)
        # Fill unmapped values with False
        series = series.where(series.notna(), False)
    else:
        series = df[field_name]

    return series.astype('bool')


# Convert field to integer type
def _convert_to_integer(df, field_name):
    numeric_series = pd.to_numeric(df[field_name], errors='coerce')
    has_nan = numeric_series.isna().any()

    if has_nan:
        return numeric_series.astype('Int64'), True
    else:
        return numeric_series.astype('int64'), False


# Convert field data type
def type_conversion(df, field_name, target_type, field_description=None):
    # Validate field exists
    if field_name not in df.columns:
        return df  # Silently skip if field not found

    # Get original type
    original_type = str(df[field_name].dtype)

    # Check if conversion is needed
    if original_type == target_type or original_type == target_type.replace('64', '32'):
        return df

    # Perform conversion
    try:
        if target_type == 'datetime64[ns]':
            df[field_name] = pd.to_datetime(df[field_name], errors='coerce')

        elif target_type == 'category':
            # Convert to object for better compatibility with statistical calculations
            df[field_name] = df[field_name].astype('object')

        elif target_type == 'bool':
            df[field_name] = _convert_to_boolean(df, field_name)

        elif target_type == 'float64':
            df[field_name] = pd.to_numeric(df[field_name], errors='coerce').astype('float64')

        elif target_type == 'int64':
            converted_series, use_nullable = _convert_to_integer(df, field_name)
            df[field_name] = converted_series

        elif target_type == 'object':
            df[field_name] = df[field_name].astype('object')

        else:
            # Generic conversion for other types
            df[field_name] = df[field_name].astype(target_type)

    except Exception:
        # Silently handle conversion errors
        pass

    return df

# Print categorical variable summary information
def print_categorical_summary(df, col):
    if col not in df.columns:
        print(f"\nError: Column '{col}' not found in DataFrame")
        return

    unique_count = df[col].nunique()
    null_count = df[col].isnull().sum()
    null_pct = (null_count / len(df)) * 100

    print(f"\n{col}:")
    print(f"  - Unique values: {unique_count:,}")
    print(f"  - Missing values: {null_count:,} ({null_pct:.2f}%)")

    # Determine number of top values to show based on cardinality
    if unique_count <= 20:
        top_n = 10
        label = "Top values"
    elif unique_count <= 100:
        top_n = 5
        label = "Top 5 values"
    else:
        top_n = 1
        label = "Most frequent"

    # Display top values
    value_counts = df[col].value_counts().head(top_n)
    print(f"  - {label}:")
    for val, count in value_counts.items():
        pct = (count / len(df)) * 100
        print(f"    * {val}: {count:,} ({pct:.1f}%)")

    # Warn about high cardinality
    if unique_count > 100:
        print(f"  - Note: High cardinality ({unique_count:,} unique values) - "
              f"consider grouping for analysis")

# Normalize axes objects to list format
def _normalize_axes(axes, n_plots):
    if n_plots == 1:
        return [axes] if not isinstance(axes, (list, np.ndarray)) else [axes[0]]
    elif hasattr(axes, 'flatten'):
        return axes.flatten().tolist()
    else:
        return [axes] if isinstance(axes, list) else list(axes)


# Plot numeric variable distribution histogram
def _plot_numeric_distribution(ax, df, col):
    df[col].hist(bins=50, ax=ax, edgecolor='black')
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.grid(True, alpha=0.3)


# Plot boolean variable distribution bar chart
def _plot_boolean_distribution(ax, df, col):
    value_counts = df[col].value_counts()

    # Use different colors for True/False
    colors = ['#1f77b4' if val else '#ff7f0e' for val in value_counts.index]
    value_counts.plot(kind='bar', ax=ax, edgecolor='black', color=colors)
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=0)
    ax.grid(True, alpha=0.3, axis='y')

    # Add percentage labels on bars
    total = len(df[col].dropna())
    for i, (val, count) in enumerate(value_counts.items()):
        pct = (count / total) * 100 if total > 0 else 0
        ax.text(i, count, f'{count:,}\n({pct:.1f}%)',
               ha='center', va='bottom', fontsize=9)


# Plot categorical variable distribution bar chart
def _plot_categorical_distribution(ax, df, col):
    unique_count = df[col].nunique()

    # Show all values if cardinality is low, otherwise top 10
    if unique_count <= 20:
        value_counts = df[col].value_counts()
    else:
        value_counts = df[col].value_counts().head(10)

    value_counts.plot(kind='bar', ax=ax, edgecolor='black')
    ax.set_title(f'Distribution of {col}')
    ax.set_xlabel(col)
    ax.set_ylabel('Frequency')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(True, alpha=0.3, axis='y')


# Plot variable distributions
def plot_distributions(df, columns, var_type='numeric'):
    if not columns or len(columns) == 0:
        print("Warning: No columns provided for plotting")
        return

    # Validate columns exist
    valid_columns = [col for col in columns if col in df.columns]
    if len(valid_columns) != len(columns):
        missing = set(columns) - set(valid_columns)
        print(f"Warning: Columns not found: {missing}")

    if not valid_columns:
        print("Error: No valid columns to plot")
        return

    # Calculate subplot grid dimensions
    n_cols = min(3, len(valid_columns))
    n_rows = (len(valid_columns) + n_cols - 1) // n_cols

    # Create subplots
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes_list = _normalize_axes(axes, len(valid_columns))

    # Plot each column
    plot_functions = {
        'numeric': _plot_numeric_distribution,
        'boolean': _plot_boolean_distribution,
        'categorical': _plot_categorical_distribution
    }

    plot_func = plot_functions.get(var_type, _plot_numeric_distribution)

    for idx, col in enumerate(valid_columns):
        plot_func(axes_list[idx], df, col)

    # Hide unused subplots
    for idx in range(len(valid_columns), len(axes_list)):
        axes_list[idx].set_visible(False)

    plt.tight_layout()
    plt.show()


# Get column types (numeric, boolean, categorical)
def get_column_types(df, exclude_cols=None):
    exclude_cols = exclude_cols or []
    numeric_cols = [col for col in df.select_dtypes(include=[np.number]).columns.tolist()
                     if col not in exclude_cols]
    bool_cols = df.select_dtypes(include=[bool]).columns.tolist()
    categorical_cols = [col for col in df.select_dtypes(include=['object']).columns.tolist()
                        if col not in exclude_cols]
    return numeric_cols, bool_cols, categorical_cols


# Categorize hour into time of day periods
def categorize_time_of_day(hour):
    if pd.isna(hour):
        return 'Unknown'
    elif 0 <= hour < 6:
        return 'Night'
    elif 6 <= hour < 12:
        return 'Morning'
    elif 12 <= hour < 18:
        return 'Afternoon'
    else:
        return 'Evening'


# Check for duplicate values in specified column
def check_duplicates(df, id_column='SeqID', top_n=5):
    counts = df[id_column].value_counts()
    duplicates = counts[counts > 1].sort_values(ascending=False)

    if not duplicates.empty:
        print(f"  - {id_column} values with duplicates: {len(duplicates)}")
        print(f"  - Top {top_n} duplicate {id_column} counts (descending):")
        print(duplicates.head(top_n).to_string())
    else:
        print(f"  - No duplicate {id_column} values found")


# Check for missing values in dataframe
def check_missing_values(df):
    missing_cols = df.columns[df.isnull().any()].tolist()
    if not missing_cols:
        print("  - No columns with missing values found")
        return

    missing_data = [(col, df[col].isnull().sum()) for col in missing_cols]
    missing_data.sort(key=lambda x: x[1], reverse=True)
    for col, missing_count in missing_data:
        missing_pct = (missing_count / df.shape[0]) * 100
        print(f"  - {col}: {missing_count:,} ({missing_pct:.2f}%)")

# ============================================================================
# SECTION 1: INTRODUCTION
# ============================================================================
print_step_header("1", "Introduction")

print("""
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
""")

# ============================================================================
# SECTION 2: BUSINESS UNDERSTANDING
# ============================================================================
print_step_header("2", "Business Understanding")

# ============================================================================
# 2.1: Determine business objectives
# ============================================================================
print_step_header("2.1", "Determine business objectives")

print("""
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
""")

# ============================================================================
# 2.2: Assess situation
# ============================================================================
print_step_header("2.2", "Assess situation")

print("""
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
""")

# ============================================================================
# 2.3: Determine data mining goals
# ============================================================================
print_step_header("2.3", "Determine data mining goals")

print("""
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
""")

# ============================================================================
# 2.4: Produce project plan
# ============================================================================
print_step_header("2.4", "Produce project plan")

print("""
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
""")

# ============================================================================
# SECTION 3: DATA UNDERSTANDING
# ============================================================================
print_step_header("3", "Data Understanding")

# ============================================================================
# 3.1: Collect initial data
# ============================================================================
print_step_header("3.1", "Collect initial data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
df = None  # Main DataFrame to store traffic violations data

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Dataset: Traffic Violations dataset from Montgomery County
# Data Source: https://data.montgomerycountymd.gov/Public-Safety/Traffic-Violations/4mse-ku6q/about_data

# Set random seeds for reproducibility
random.seed(2025)
np.random.seed(2025)

# Load the dataset
df = pd.read_csv(csv_path)

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nDataset loaded: {len(df):,} records × {len(df.columns)} columns from {csv_path}")

# ============================================================================
# 3.2: Describe data
# ============================================================================
print_step_header("3.2", "Describe data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
# Field definitions: (field_name, target_type, field_description)
FIELD_DEFINITIONS = [
    ('SeqID',                    'object',        "Unique sequence identifier for the traffic stop record"),
    ('Date Of Stop',             'datetime64[ns]', "Date when the traffic stop occurred"),
    ('Time Of Stop',             'object',        "Time when the traffic stop occurred"),
    ('Agency',                   'object',        "Law enforcement agency that conducted the stop"),
    ('SubAgency',                'object',        "Sub-agency or division that conducted the stop"),
    ('Description',              'object',        "Description of the traffic violation incident"),
    ('Location',                 'object',        "Location where the traffic stop occurred"),
    ('Latitude',                 'float64',       "Latitude coordinate of the stop location"),
    ('Longitude',                'float64',       "Longitude coordinate of the stop location"),
    ('Accident',                 'bool',          "Whether an accident was involved"),
    ('Belts',                    'bool',          "Whether seat belts were used"),
    ('Personal Injury',          'bool',          "Whether there was personal injury"),
    ('Property Damage',          'bool',          "Whether there was property damage"),
    ('Fatal',                    'bool',          "Whether the incident was fatal"),
    ('Commercial License',       'bool',          "Whether the driver had a commercial license"),
    ('HAZMAT',                   'bool',          "Whether hazardous materials were involved"),
    ('Commercial Vehicle',       'bool',          "Whether the vehicle was commercial"),
    ('Alcohol',                  'bool',          "Whether alcohol was involved"),
    ('Work Zone',                'bool',          "Whether the incident occurred in a work zone"),
    ('Search Conducted',         'bool',          "Whether a search was conducted"),
    ('Search Disposition',       'object',        "Disposition of the search conducted"),
    ('Search Outcome',           'object',        "Outcome of the search conducted"),
    ('Search Reason',            'object',        "Reason for conducting the search"),
    ('Search Reason For Stop',   'object',        "Reason for the stop that led to search"),
    ('Search Type',              'object',        "Type of search conducted"),
    ('Search Arrest Reason',     'object',        "Reason for arrest if search led to arrest"),
    ('State',                    'object',        "US state where the stop occurred"),
    ('VehicleType',              'object',        "Type of vehicle involved"),
    ('Year',                     'int64',         "Year of the vehicle"),
    ('Make',                     'object',        "Make/manufacturer of the vehicle"),
    ('Model',                    'object',        "Model of the vehicle"),
    ('Color',                    'object',        "Color of the vehicle"),
    ('Violation Type',           'object',        "Type of traffic violation"),
    ('Charge',                   'object',        "Charge filed against the driver"),
    ('Article',                  'object',        "Legal article or statute related to the violation"),
    ('Contributed To Accident',  'bool',          "Whether the violation contributed to an accident"),
    ('Race',                     'object',        "Race/ethnicity of the driver"),
    ('Gender',                   'object',        "Gender of the driver"),
    ('Driver City',              'object',        "City where the driver resides"),
    ('Driver State',             'object',        "US state where the driver resides"),
    ('DL State',                 'object',        "State that issued the driver's license"),
    ('Arrest Type',              'object',        "Type of arrest made during the stop"),
    ('Geolocation',              'object',        "Geographic location information of the stop"),
]

numeric_cols      = None  # List of numeric column names
categorical_cols  = None  # List of categorical column names
boolean_cols      = None  # List of boolean column names
col_unique_counts = None  # List of tuples (column_name, unique_count) for categorical columns

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Convert field types
for idx, (field_name, target_type, field_description) in enumerate(FIELD_DEFINITIONS, 1):
    df = type_conversion(df, field_name, target_type, field_description)

# Get column types for statistics
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
boolean_cols = df.select_dtypes(include=['bool']).columns.tolist()

if categorical_cols:
    col_unique_counts = [(col, df[col].nunique()) for col in categorical_cols]
    col_unique_counts.sort(key=lambda x: x[1], reverse=True)

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
numeric_count = len(numeric_cols) if numeric_cols else 0
categorical_count = len(categorical_cols) if categorical_cols else 0
boolean_count = len(boolean_cols) if boolean_cols else 0
print(f"\nData description: {numeric_count} numeric, {categorical_count} categorical, {boolean_count} boolean columns")

# ============================================================================
# 3.3: Explore data
# ============================================================================
print_step_header("3.3", "Explore data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
numeric_cols     = None  # List of numeric column names
bool_cols        = None  # List of boolean column names
categorical_cols = None  # List of categorical column names

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
numeric_cols, bool_cols, categorical_cols = get_column_types(df, exclude_cols=['SeqID'])

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nData exploration: {len(numeric_cols)} numeric, {len(bool_cols)} boolean, {len(categorical_cols)} categorical variables visualized")

# Numeric Variable Distributions
if numeric_cols:
    plot_distributions(df, numeric_cols, var_type='numeric')

# Boolean Variable Distributions
if bool_cols:
    plot_distributions(df, bool_cols, var_type='boolean')

# Categorical Variable Distributions (showing top 10)
if categorical_cols:
    for col in categorical_cols[:10]:
        print_categorical_summary(df, col)

# ============================================================================
# 3.4: Verify data quality
# ============================================================================
# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
dup_count    = None  # Number of duplicate SeqID values
missing_info = None  # Missing value information

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Count duplicates
dup_count = df['SeqID'].duplicated().sum()

# Get missing value counts
missing_counts = df.isnull().sum()
missing_cols = missing_counts[missing_counts > 0]

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nData quality: {dup_count:,} duplicates, {len(missing_cols)} columns with missing values")

# Visualize data quality issues
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Duplicate check visualization
dup_data = ['Unique', 'Duplicates']
dup_values = [len(df) - dup_count, dup_count]
axes[0].bar(dup_data, dup_values, color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Duplicate Records Check', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(dup_values):
    pct = val / len(df) * 100
    axes[0].text(i, val, f'{val:,}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=10)

# Missing values visualization
if len(missing_cols) > 0:
    top_missing = missing_cols.nlargest(10)
    axes[1].barh(range(len(top_missing)), top_missing.values, color='#e67e22', edgecolor='black', alpha=0.7)
    axes[1].set_yticks(range(len(top_missing)))
    axes[1].set_yticklabels(top_missing.index, fontsize=9)
    axes[1].set_xlabel('Missing Count', fontsize=12)
    axes[1].set_title('Top 10 Columns with Missing Values', fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='x')
    axes[1].invert_yaxis()
    for i, val in enumerate(top_missing.values):
        pct = val / len(df) * 100
        axes[1].text(val, i, f' {val:,} ({pct:.1f}%)', va='center', fontsize=9)
else:
    axes[1].text(0.5, 0.5, 'No Missing Values', ha='center', va='center', 
                transform=axes[1].transAxes, fontsize=14, fontweight='bold')
    axes[1].set_title('Missing Values Check', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 4. Classification by Decision Tree
# ============================================================================
print_step_header("4", "Classification by Decision Tree")

# ============================================================================
# 4.1. Data Preparation for Classification
# ============================================================================
print_step_header("4.1", "Data Preparation for Classification")

# ============================================================================
# 4.1.1. Select data
# ============================================================================
print_step_header("4.1.1", "Select data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
df_classify             = None  # DataFrame for classification
classification_features = None  # List of features for classification
target_variable         = None  # Target variable

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
df_classify = df.copy()

# Select relevant attributes for classification
classification_features = [
    'Date Of Stop', 'Time Of Stop',                               # Temporal features
    'SubAgency', 'Location', 'Latitude', 'Longitude',             # Location features
    'Accident', 'Personal Injury', 'Property Damage', 'Fatal',    # Accident-related
    'Belts', 'Alcohol', 'Work Zone',                              # Risk factors
    'Commercial License', 'Commercial Vehicle', 'HAZMAT',         # Driver/vehicle type
    'State', 'VehicleType', 'Year', 'Make', 'Color',              # Vehicle characteristics
    'Contributed To Accident',                                    # Accident contribution
    'Race', 'Gender'                                              # Demographics
]

target_variable = 'Violation Type'  # Class variable

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nClassification setup: {len(classification_features)} features selected for predicting {target_variable}")

# Visualize feature selection
fig, ax = plt.subplots(figsize=(10, 6))
feature_categories = ['Temporal', 'Location', 'Accident', 'Risk', 'Driver/Vehicle', 'Demographics']
feature_counts = [2, 4, 4, 3, 6, 2]
ax.barh(feature_categories, feature_counts, color='steelblue', edgecolor='black', alpha=0.7)
ax.set_xlabel('Number of Features', fontsize=12)
ax.set_title(f'Feature Selection for Classification (Total: {len(classification_features)})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
for i, val in enumerate(feature_counts):
    ax.text(val, i, f' {val}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 4.1.2. Clean data
# ============================================================================
print_step_header("4.1.2", "Clean data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
min_instances    = None  # Minimum instances required for a class
class_counts     = None  # Counts of each class
valid_violations = None  # List of valid violation types
original_size    = None  # Original size of the dataset

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Remove rows with missing target variable
original_size = len(df_classify)
df_classify = df_classify.dropna(subset=[target_variable])

# Keep only top violation types to make the problem more manageable
# Filter to keep violation types with at least 1000 instances
min_instances    = 1000
class_counts     = df_classify[target_variable].value_counts()
valid_violations = class_counts[class_counts >= min_instances].index.tolist()
df_classify = df_classify[df_classify[target_variable].isin(valid_violations)]

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nData cleaning: {original_size:,} → {len(df_classify):,} rows, {len(valid_violations)} classes retained")

# Visualize data cleaning impact
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Before/After comparison
axes[0].bar(['Before', 'After'], [original_size, len(df_classify)], 
            color=['#95a5a6', '#2ecc71'], edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Number of Records', fontsize=12)
axes[0].set_title('Data Cleaning Impact', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate([original_size, len(df_classify)]):
    axes[0].text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Class distribution after filtering
top_classes = df_classify[target_variable].value_counts().head(10)
axes[1].barh(range(len(top_classes)), top_classes.values, color='#3498db', edgecolor='black', alpha=0.7)
axes[1].set_yticks(range(len(top_classes)))
axes[1].set_yticklabels(top_classes.index, fontsize=9)
axes[1].set_xlabel('Count', fontsize=12)
axes[1].set_title(f'Top 10 Violation Types (Total: {len(valid_violations)} classes)', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# ============================================================================
# 4.1.3. Construct data
# ============================================================================
print_step_header("4.1.3", "Construct data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
current_year = None  # Current year for age calculation

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------

# Extract hour from Time Of Stop
df_classify['Hour'] = pd.to_datetime(df_classify['Time Of Stop'], format='%H:%M:%S', errors='coerce').dt.hour

# Extract date features
df_classify['Month'] = df_classify['Date Of Stop'].dt.month
df_classify['DayOfWeek'] = df_classify['Date Of Stop'].dt.dayofweek
df_classify['IsWeekend'] = (df_classify['DayOfWeek'] >= 5).astype(int)

df_classify['TimeOfDay'] = df_classify['Hour'].apply(categorize_time_of_day)

current_year = 2024
df_classify['VehicleAge'] = current_year - df_classify['Year'].fillna(current_year)

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print("\nFeature engineering: Created 6 new temporal and vehicle features")

# Visualize engineered features
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Hour distribution
df_classify['Hour'].hist(bins=24, ax=axes[0], edgecolor='black', color='#3498db', alpha=0.7)
axes[0].set_title('Hour Distribution', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Month distribution
df_classify['Month'].hist(bins=12, ax=axes[1], edgecolor='black', color='#2ecc71', alpha=0.7)
axes[1].set_title('Month Distribution', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

# DayOfWeek distribution
df_classify['DayOfWeek'].hist(bins=7, ax=axes[2], edgecolor='black', color='#e74c3c', alpha=0.7)
axes[2].set_title('Day of Week Distribution', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Day (0=Mon, 6=Sun)')
axes[2].set_ylabel('Frequency')
axes[2].grid(True, alpha=0.3)

# IsWeekend distribution
weekend_counts = df_classify['IsWeekend'].value_counts()
axes[3].bar(['Weekday', 'Weekend'], [weekend_counts.get(0, 0), weekend_counts.get(1, 0)], 
            color=['#3498db', '#e67e22'], edgecolor='black', alpha=0.7)
axes[3].set_title('Weekend vs Weekday', fontsize=11, fontweight='bold')
axes[3].set_ylabel('Frequency')
axes[3].grid(True, alpha=0.3, axis='y')

# TimeOfDay distribution
time_counts = df_classify['TimeOfDay'].value_counts()
axes[4].bar(time_counts.index, time_counts.values, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[4].set_title('Time of Day Distribution', fontsize=11, fontweight='bold')
axes[4].set_ylabel('Frequency')
axes[4].tick_params(axis='x', rotation=45)
axes[4].grid(True, alpha=0.3, axis='y')

# VehicleAge distribution
df_classify['VehicleAge'].hist(bins=30, ax=axes[5], edgecolor='black', color='#1abc9c', alpha=0.7)
axes[5].set_title('Vehicle Age Distribution', fontsize=11, fontweight='bold')
axes[5].set_xlabel('Age (years)')
axes[5].set_ylabel('Frequency')
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 4.1.4. Integrate data
# ============================================================================
print_step_header("4.1.4", "Integrate data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
final_features = None  # Final list of features for model

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Select final features for classification
final_features = [
    'Hour', 'Month', 'DayOfWeek', 'IsWeekend',  # Temporal
    'Latitude', 'Longitude',  # Geographic
    'VehicleAge',  # Vehicle
    'Accident', 'Personal Injury', 'Property Damage', 'Fatal',  # Accident
    'Belts', 'Alcohol', 'Work Zone',  # Risk factors
    'Commercial License', 'Commercial Vehicle', 'HAZMAT',  # Driver/vehicle type
    'Contributed To Accident',  # Accident contribution
    'SubAgency', 'VehicleType', 'Gender', 'Race', 'TimeOfDay'  # Categorical
]

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nFeature integration: {len(final_features)} features integrated across 5 categories")

# Visualize feature integration
fig, ax = plt.subplots(figsize=(10, 6))
categories = ['Temporal (4)', 'Geographic (2)', 'Vehicle (1)', 'Accident (4)', 
              'Risk Factors (3)', 'Driver/Vehicle Type (3)', 'Accident Contrib (1)', 'Categorical (5)']
counts = [4, 2, 1, 4, 3, 3, 1, 5]
colors = plt.cm.viridis(np.linspace(0, 1, len(categories)))
ax.barh(categories, counts, color=colors, edgecolor='black', alpha=0.7)
ax.set_xlabel('Number of Features', fontsize=12)
ax.set_title(f'Feature Integration by Category (Total: {len(final_features)})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
for i, val in enumerate(counts):
    ax.text(val, i, f' {val}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 4.1.5. Format data
# ============================================================================
print_step_header("4.1.5", "Format data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
categorical_features_classify = None  # List of categorical features
label_encoders_classify       = None  # Dictionary to store label encoders
numeric_features_classify     = None  # List of numeric features
boolean_features_classify     = None  # List of boolean features
encoded_features_classify     = None  # List of encoded feature names
all_features_classify         = None  # Combined list of all features
X_classify                    = None  # Feature matrix
y_classify                    = None  # Target vector

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Encode categorical variables

categorical_features_classify = ['SubAgency', 'VehicleType', 'Gender', 'Race', 'TimeOfDay']
label_encoders_classify = {}

for feature in categorical_features_classify:
    le = LabelEncoder()
    df_classify[feature] = df_classify[feature].fillna('Unknown')
    df_classify[feature + '_encoded'] = le.fit_transform(df_classify[feature].astype(str))
    label_encoders_classify[feature] = le

# Prepare feature matrix
numeric_features_classify = [
    'Hour', 'Month', 'DayOfWeek', 'IsWeekend',
    'Latitude', 'Longitude', 'VehicleAge'
]

boolean_features_classify = [
    'Accident', 'Personal Injury', 'Property Damage', 'Fatal',
    'Belts', 'Alcohol', 'Work Zone',
    'Commercial License', 'Commercial Vehicle', 'HAZMAT',
    'Contributed To Accident'
]

encoded_features_classify = [feat + '_encoded' for feat in categorical_features_classify]

all_features_classify = numeric_features_classify + boolean_features_classify + encoded_features_classify

# Create feature matrix and target
X_classify = df_classify[all_features_classify].copy()
y_classify = df_classify[target_variable].copy()

# Convert boolean to int
for col in boolean_features_classify:
    X_classify[col] = X_classify[col].astype(int)

# Handle missing values
X_classify = X_classify.fillna(0)

# Apply stratified sampling if dataset is too large (requirement: sample to 10,000)
target_sample_size_classify = 10000
if len(df_classify) > target_sample_size_classify:
    sample_fraction_classify = target_sample_size_classify / len(df_classify)
    df_classify = df_classify.groupby(target_variable, group_keys=False).apply(
        lambda x: x.sample(frac=sample_fraction_classify, random_state=2025)
    )
    # Recreate feature matrix and target after sampling
    X_classify = df_classify[all_features_classify].copy()
    y_classify = df_classify[target_variable].copy()
    # Convert boolean to int again after sampling
    for col in boolean_features_classify:
        X_classify[col] = X_classify[col].astype(int)
    # Handle missing values again
    X_classify = X_classify.fillna(0)

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nData formatting: {X_classify.shape[0]:,} samples × {X_classify.shape[1]} features, {y_classify.nunique()} classes prepared")

# Visualize data formatting summary
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature type breakdown
feature_types = ['Numeric (7)', 'Boolean (11)', 'Encoded Cat (5)']
type_counts = [7, 11, 5]
axes[0].bar(feature_types, type_counts, color=['#3498db', '#2ecc71', '#e67e22'], edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title(f'Feature Types (Total: {X_classify.shape[1]})', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(type_counts):
    axes[0].text(i, val, f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Class distribution
class_dist = y_classify.value_counts().head(10)
axes[1].barh(range(len(class_dist)), class_dist.values, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[1].set_yticks(range(len(class_dist)))
axes[1].set_yticklabels(class_dist.index, fontsize=9)
axes[1].set_xlabel('Count', fontsize=12)
axes[1].set_title(f'Top 10 Classes (Total: {y_classify.nunique()})', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='x')
axes[1].invert_yaxis()

plt.tight_layout()
plt.show()

# ============================================================================
# 4.2. Modelling
# ============================================================================
print_step_header("4.2", "Modelling")

# ============================================================================
# 4.2.1. Select modeling techniques
# ============================================================================
print_step_header("4.2.1", "Select modeling techniques")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print("\nModel selection: Decision Tree Classifier (interpretable, suitable for classification)")

# ============================================================================
# 4.2.2. Generate test design
# ============================================================================
print_step_header("4.2.2", "Generate test design")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
X_train = None  # Training features
X_test  = None  # Testing features
y_train = None  # Training target
y_test  = None  # Testing target

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_classify, y_classify, test_size=0.3, random_state=2025, stratify=y_classify
)

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nTest design: Train/Test split 70%/30% with stratification")

# Visualize train/test split
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Sample count comparison
axes[0].bar(['Training', 'Testing'], [len(X_train), len(X_test)], 
            color=['#3498db', '#e67e22'], edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Number of Samples', fontsize=12)
axes[0].set_title('Train/Test Split', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate([len(X_train), len(X_test)]):
    pct = val / (len(X_train) + len(X_test)) * 100
    axes[0].text(i, val, f'{val:,}\n({pct:.0f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Class distribution in train vs test
train_dist = y_train.value_counts().head(5)
test_dist = y_test.value_counts().head(5)
x = np.arange(len(train_dist))
width = 0.35
axes[1].bar(x - width/2, train_dist.values, width, label='Train', color='#3498db', edgecolor='black', alpha=0.7)
axes[1].bar(x + width/2, test_dist.values, width, label='Test', color='#e67e22', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Violation Type', fontsize=12)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Top 5 Classes Distribution', fontsize=13, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(train_dist.index, rotation=45, ha='right', fontsize=9)
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# 4.2.3. Build model
# ============================================================================
print_step_header("4.2.3", "Build model")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
dt_classifier      = None  # Decision Tree Classifier object
dt_cv              = None  # Decision Tree for CV
y_pred_train       = None  # Predictions on training set
y_pred_test        = None  # Predictions on testing set
train_accuracy     = None  # Accuracy on training set
test_accuracy      = None  # Accuracy on testing set
cv_scores          = None  # Cross-validation scores
cm                 = None  # Confusion matrix
feature_importance = None  # Feature importance dataframe

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Create and train decision tree
dt_classifier = DecisionTreeClassifier(
    max_depth=5,  # Limit depth for interpretability
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=2025
)

dt_classifier.fit(X_train, y_train)

# Make predictions
y_pred_train = dt_classifier.predict(X_train)
y_pred_test = dt_classifier.predict(X_test)

# Evaluate performance
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

# Cross-validation (required by project specifications)
dt_cv = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=50,
    min_samples_leaf=20,
    random_state=2025
)
cv_scores = cross_val_score(dt_cv, X_classify, y_classify, cv=5, scoring='accuracy')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred_test)

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': all_features_classify,
    'Importance': dt_classifier.feature_importances_
}).sort_values('Importance', ascending=False)

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nModel Performance: Training accuracy: {train_accuracy:.2%} | Test accuracy: {test_accuracy:.2%}")

# Visualize top features
plt.figure(figsize=(10, 6))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'], 
         color='steelblue', edgecolor='black')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Importance', fontsize=12)
plt.title('Top 10 Feature Importances', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
for i, (idx, row) in enumerate(top_features.iterrows()):
    plt.text(row['Importance'], i, f" {row['Importance']:.3f}", 
             va='center', fontsize=9)
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# ============================================================================
# 4.2.4. Assess model
# ============================================================================
print_step_header("4.2.4", "Assess model")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
# Plot decision tree (simplified view)
plt.figure(figsize=(25, 15))
plot_tree(dt_classifier,
          feature_names=all_features_classify,
          class_names=dt_classifier.classes_,
          filled=True,
          fontsize=8,
          max_depth=3)  # Only show top 3 levels for readability
plt.title('Decision Tree for Traffic Violation Classification (Top 3 Levels)', fontsize=16)
plt.tight_layout()
plt.show()

# Plot feature importance
plt.figure(figsize=(12, 6))
top_features = feature_importance.head(15)
plt.barh(top_features['Feature'], top_features['Importance'], edgecolor='black')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 15 Feature Importances for Violation Type Classification')
plt.gca().invert_yaxis()
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.show()

# Model built successfully - decision tree and feature importance visualized above

# ============================================================================
# 4.3. Evaluation
# ============================================================================
print_step_header("4.3", "Evaluation")

# ============================================================================
# 4.3.1. Evaluate results
# ============================================================================
print_step_header("4.3.1", "Evaluate results")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nModel Performance: Train {train_accuracy:.2%} | Test {test_accuracy:.2%} | CV {cv_scores.mean():.2%} (±{cv_scores.std() * 2:.2%})")

# Visualize performance metrics
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Accuracy comparison
axes[0].bar(['Train', 'Test', 'CV Mean'], 
            [train_accuracy, test_accuracy, cv_scores.mean()],
            color=['#2ecc71', '#3498db', '#9b59b6'], edgecolor='black', alpha=0.7)
axes[0].errorbar(2, cv_scores.mean(), yerr=cv_scores.std() * 2, 
                 fmt='none', color='red', capsize=5, capthick=2)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Comparison', fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate([train_accuracy, test_accuracy, cv_scores.mean()]):
    axes[0].text(i, val + 0.02, f'{val:.2%}', ha='center', fontsize=10, fontweight='bold')

# Cross-validation scores distribution
axes[1].boxplot(cv_scores, vert=True, patch_artist=True,
                boxprops=dict(facecolor='#9b59b6', alpha=0.7, linewidth=1.5),
                medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Accuracy', fontsize=12)
axes[1].set_title(f'Cross-Validation Scores\n(Mean: {cv_scores.mean():.2%} ± {cv_scores.std() * 2:.2%})', 
                  fontsize=13, fontweight='bold')
axes[1].set_xticklabels(['CV Scores'])
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# 4.3.2. Interpret results
# ============================================================================
print_step_header("4.3.2", "Interpret results")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nConfusion Matrix: {cm.shape[0]}×{cm.shape[1]} classification results")

# Prepare confusion matrix data
cm_df = pd.DataFrame(cm, 
                     index=dt_classifier.classes_, 
                     columns=dt_classifier.classes_)

# Calculate percentages for better visualization
cm_percent = cm_df.div(cm_df.sum(axis=1), axis=0) * 100

# Visualize confusion matrix
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Confusion matrix (counts)
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=axes[0],
            xticklabels=dt_classifier.classes_,
            yticklabels=dt_classifier.classes_,
            cbar_kws={'label': 'Count'})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=13, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=11)
axes[0].set_xlabel('Predicted Label', fontsize=11)

# Confusion matrix (percentages)
sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='Blues', ax=axes[1],
            xticklabels=dt_classifier.classes_,
            yticklabels=dt_classifier.classes_,
            cbar_kws={'label': 'Percentage (%)'})
axes[1].set_title('Confusion Matrix (Percentages)', fontsize=13, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=11)
axes[1].set_xlabel('Predicted Label', fontsize=11)

plt.tight_layout()
plt.show()

# ============================================================================
# 4.3.3. Review of process
# ============================================================================
print_step_header("4.3.3", "Review of process")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print("\nProcess review: Complete workflow from data preparation to model evaluation")

# Visualize process workflow
fig, ax = plt.subplots(figsize=(12, 6))
steps = ['Data\nSelection', 'Data\nCleaning', 'Feature\nEngineering', 
         'Feature\nIntegration', 'Data\nFormatting', 'Model\nTraining', 'Evaluation']
status = [1, 1, 1, 1, 1, 1, 1]  # All completed
colors = ['#2ecc71' if s == 1 else '#95a5a6' for s in status]

y_pos = np.arange(len(steps))
ax.barh(y_pos, status, color=colors, edgecolor='black', alpha=0.7)
ax.set_yticks(y_pos)
ax.set_yticklabels(steps, fontsize=11)
ax.set_xlabel('Status', fontsize=12)
ax.set_title('Classification Process Review', fontsize=14, fontweight='bold')
ax.set_xlim([0, 1.2])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not Started', 'Completed'])
ax.invert_yaxis()

for i, (step, s) in enumerate(zip(steps, status)):
    ax.text(1.05, i, '✓ Completed', va='center', fontsize=10, fontweight='bold', color='#2ecc71')

plt.tight_layout()
plt.show()

# ============================================================================
# 4.3.4. Determine next steps
# ============================================================================
print_step_header("4.3.4", "Determine next steps")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nClassification complete: {y_classify.nunique()} classes, {len(all_features_classify)} features, Test accuracy {test_accuracy:.2%}")

# Visualize final summary
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Model performance summary
metrics = ['Train Acc', 'Test Acc', 'CV Mean']
values = [train_accuracy, test_accuracy, cv_scores.mean()]
colors_perf = ['#2ecc71', '#3498db', '#9b59b6']
axes[0].bar(metrics, values, color=colors_perf, edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Performance Summary', fontsize=13, fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(values):
    axes[0].text(i, val + 0.02, f'{val:.2%}', ha='center', fontsize=10, fontweight='bold')

# Dataset summary
dataset_info = ['Total\nRecords', 'Features', 'Classes']
dataset_values = [len(X_classify), len(all_features_classify), y_classify.nunique()]
axes[1].bar(dataset_info, dataset_values, color='#e67e22', edgecolor='black', alpha=0.7)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Dataset Summary', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(dataset_values):
    axes[1].text(i, val, f'{val:,}' if val > 100 else f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Top 5 important features
top_5_features = feature_importance.head(5)
axes[2].barh(range(len(top_5_features)), top_5_features['Importance'], color='#1abc9c', edgecolor='black', alpha=0.7)
axes[2].set_yticks(range(len(top_5_features)))
axes[2].set_yticklabels(top_5_features['Feature'], fontsize=9)
axes[2].set_xlabel('Importance', fontsize=12)
axes[2].set_title('Top 5 Important Features', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='x')
axes[2].invert_yaxis()

plt.tight_layout()
plt.show()

# ============================================================================
# 5. Clustering by KMeans
# ============================================================================
print_step_header("5", "Clustering by KMeans")

# ============================================================================
# 5.1. Data Preparation for Clustering
# ============================================================================
print_step_header("5.1", "Data Preparation for Clustering")

df_cluster = df.copy()

# ============================================================================
# 5.1.1. Select Data
# ============================================================================

print_step_header("5.1.1", "Select Data - Select and justify features for clustering")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
clustering_features = None  # Features selected for clustering

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Select relevant attributes for clustering
# Focus on features that can reveal patterns in violations
clustering_features = [
    'Date Of Stop', 'Time Of Stop',  # Temporal
    'Latitude', 'Longitude',  # Geographic
    'Accident', 'Personal Injury', 'Property Damage', 'Fatal',  # Severity
    'Alcohol', 'Work Zone',  # Risk factors
    'Commercial Vehicle', 'HAZMAT',  # Vehicle type
    'Year',  # Vehicle age
    'VehicleType', 'SubAgency', 'Gender', 'Race'  # Categorical
]

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nClustering setup: {len(clustering_features)} features selected for pattern discovery")

# Visualize feature selection for clustering
fig, ax = plt.subplots(figsize=(10, 6))
feature_cats = ['Temporal', 'Geographic', 'Severity', 'Risk', 'Vehicle', 'Safety', 'Demographics', 'Enforcement']
feature_cnts = [2, 2, 4, 2, 4, 2, 2, 1]
ax.barh(feature_cats, feature_cnts, color='#9b59b6', edgecolor='black', alpha=0.7)
ax.set_xlabel('Number of Features', fontsize=12)
ax.set_title(f'Feature Selection for Clustering (Total: {len(clustering_features)})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
for i, val in enumerate(feature_cnts):
    ax.text(val, i, f' {val}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 5.1.2. Clean Data
# ============================================================================

print_step_header("5.1.2", "Clean Data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
original_size_cluster = None  # Original size of clustering dataset

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
original_size_cluster = len(df_cluster)

# Remove rows with missing critical coordinates
df_cluster = df_cluster.dropna(subset=['Latitude', 'Longitude'])

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nData cleaning: {original_size_cluster:,} → {len(df_cluster):,} rows (removed missing coordinates)")

# Visualize data cleaning
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(['Before', 'After'], [original_size_cluster, len(df_cluster)], 
       color=['#95a5a6', '#2ecc71'], edgecolor='black', alpha=0.7)
ax.set_ylabel('Number of Records', fontsize=12)
ax.set_title('Clustering Data Cleaning', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, val in enumerate([original_size_cluster, len(df_cluster)]):
    pct = val / original_size_cluster * 100
    ax.text(i, val, f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 5.1.3. Construct Data
# ============================================================================

print_step_header("5.1.3", "Construct Data - Feature Engineering")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
current_year = None  # Current year for age calculation

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
df_cluster['Hour'] = pd.to_datetime(df_cluster['Time Of Stop'], format='%H:%M:%S', errors='coerce').dt.hour
df_cluster['Month'] = df_cluster['Date Of Stop'].dt.month
df_cluster['DayOfWeek'] = df_cluster['Date Of Stop'].dt.dayofweek
df_cluster['IsWeekend'] = (df_cluster['DayOfWeek'] >= 5).astype(int)

# Time of day
df_cluster['TimeOfDay'] = df_cluster['Hour'].apply(categorize_time_of_day)

# Vehicle age
current_year = 2025
df_cluster['VehicleAge'] = current_year - df_cluster['Year'].fillna(current_year)

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nFeature engineering: Created 6 temporal/vehicle features (mean VehicleAge={df_cluster['VehicleAge'].mean():.1f}yr)")

# Visualize engineered features for clustering
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Hour distribution
df_cluster['Hour'].hist(bins=24, ax=axes[0], edgecolor='black', color='#9b59b6', alpha=0.7)
axes[0].set_title('Hour Distribution', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Month distribution
df_cluster['Month'].hist(bins=12, ax=axes[1], edgecolor='black', color='#3498db', alpha=0.7)
axes[1].set_title('Month Distribution', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

# DayOfWeek distribution
df_cluster['DayOfWeek'].hist(bins=7, ax=axes[2], edgecolor='black', color='#2ecc71', alpha=0.7)
axes[2].set_title('Day of Week Distribution', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Day (0=Mon, 6=Sun)')
axes[2].set_ylabel('Frequency')
axes[2].grid(True, alpha=0.3)

# IsWeekend distribution
weekend_cnts = df_cluster['IsWeekend'].value_counts()
axes[3].bar(['Weekday', 'Weekend'], [weekend_cnts.get(0, 0), weekend_cnts.get(1, 0)], 
            color=['#3498db', '#e67e22'], edgecolor='black', alpha=0.7)
axes[3].set_title('Weekend vs Weekday', fontsize=11, fontweight='bold')
axes[3].set_ylabel('Frequency')
axes[3].grid(True, alpha=0.3, axis='y')

# TimeOfDay distribution
time_cnts = df_cluster['TimeOfDay'].value_counts()
axes[4].bar(time_cnts.index, time_cnts.values, color='#e74c3c', edgecolor='black', alpha=0.7)
axes[4].set_title('Time of Day Distribution', fontsize=11, fontweight='bold')
axes[4].set_ylabel('Frequency')
axes[4].tick_params(axis='x', rotation=45)
axes[4].grid(True, alpha=0.3, axis='y')

# VehicleAge distribution
df_cluster['VehicleAge'].hist(bins=30, ax=axes[5], edgecolor='black', color='#1abc9c', alpha=0.7)
axes[5].set_title('Vehicle Age Distribution', fontsize=11, fontweight='bold')
axes[5].set_xlabel('Age (years)')
axes[5].set_ylabel('Frequency')
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 5.1.4. Integrate Data
# ============================================================================

print_step_header("5.1.4", "Integrate Data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
final_features_cluster = None  # Final features for clustering

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
final_features_cluster = [
    'Hour', 'Month', 'DayOfWeek', 'IsWeekend',
    'Latitude', 'Longitude',
    'VehicleAge',
    'Accident', 'Personal Injury', 'Property Damage', 'Fatal',
    'Alcohol', 'Work Zone',
    'Commercial Vehicle', 'HAZMAT',
    'VehicleType', 'SubAgency', 'Gender', 'Race', 'TimeOfDay'
]

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nFeature integration: {len(final_features_cluster)} features across 5 categories")

# Visualize feature integration for clustering
fig, ax = plt.subplots(figsize=(10, 6))
cats = ['Temporal (4)', 'Geographic (2)', 'Vehicle (1)', 'Severity (4)', 
        'Risk (2)', 'Vehicle Type (2)', 'Categorical (5)']
cnts = [4, 2, 1, 4, 2, 2, 5]
colors_cluster = plt.cm.plasma(np.linspace(0, 1, len(cats)))
ax.barh(cats, cnts, color=colors_cluster, edgecolor='black', alpha=0.7)
ax.set_xlabel('Number of Features', fontsize=12)
ax.set_title(f'Feature Integration for Clustering (Total: {len(final_features_cluster)})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
for i, val in enumerate(cnts):
    ax.text(val, i, f' {val}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 5.1.5. Format Data
# ============================================================================

print_step_header("5.1.5", "Format Data - Encode, scale, and validate")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
categorical_features_cluster = None  # Categorical features for clustering
label_encoders_cluster       = None  # Label encoders for clustering
numeric_features_cluster     = None  # Numeric features for clustering
boolean_features_cluster     = None  # Boolean features for clustering
encoded_features_cluster     = None  # Encoded feature names
all_features_cluster         = None  # All features for clustering
X_cluster                    = None  # Feature matrix for clustering
scaler_cluster               = None  # StandardScaler object
X_cluster_scaled             = None  # Scaled feature matrix
sample_size                  = None  # Sample size for clustering
sample_indices               = None  # Indices for sampling
X_cluster_sampled            = None  # Sampled scaled features
df_cluster_sampled           = None  # Sampled dataframe

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Encode categorical variables
categorical_features_cluster = ['VehicleType', 'SubAgency', 'Gender', 'Race', 'TimeOfDay']
label_encoders_cluster = {}

for feature in categorical_features_cluster:
    le = LabelEncoder()
    df_cluster[feature] = df_cluster[feature].fillna('Unknown')
    df_cluster[feature + '_encoded'] = le.fit_transform(df_cluster[feature].astype(str))
    label_encoders_cluster[feature] = le

# Prepare feature matrix
numeric_features_cluster = [
    'Hour', 'Month', 'DayOfWeek', 'IsWeekend',
    'Latitude', 'Longitude', 'VehicleAge'
]

boolean_features_cluster = [
    'Accident', 'Personal Injury', 'Property Damage', 'Fatal',
    'Alcohol', 'Work Zone',
    'Commercial Vehicle', 'HAZMAT'
]

encoded_features_cluster = [feat + '_encoded' for feat in categorical_features_cluster]

all_features_cluster = numeric_features_cluster + boolean_features_cluster + encoded_features_cluster

X_cluster = df_cluster[all_features_cluster].copy()

# Convert boolean to int
for col in boolean_features_cluster:
    X_cluster[col] = X_cluster[col].astype(int)

# Handle missing values
X_cluster = X_cluster.fillna(0)

# Apply stratified sampling if dataset is too large (requirement: sample to 10,000)
# Do sampling BEFORE scaling to ensure consistency
target_sample_size_cluster = 10000
if len(df_cluster) > target_sample_size_cluster:
    sample_size = target_sample_size_cluster
    # Try to use Violation Type for stratification, if not available use a combination of key features
    if 'Violation Type' in df_cluster.columns:
        stratify_col = 'Violation Type'
    elif 'Accident' in df_cluster.columns:
        stratify_col = 'Accident'
    else:
        # Create a composite stratification variable
        df_cluster['_stratify'] = (
            df_cluster['Hour'].astype(str) + '_' +
            df_cluster['IsWeekend'].astype(str) + '_' +
            (df_cluster['Accident'].astype(int) if 'Accident' in df_cluster.columns else '0')
        )
        stratify_col = '_stratify'

    sample_fraction_cluster = sample_size / len(df_cluster)
    df_cluster = df_cluster.groupby(stratify_col, group_keys=False).apply(
        lambda x: x.sample(frac=sample_fraction_cluster, random_state=2025)
    )
    # Remove temporary stratification column if created
    if '_stratify' in df_cluster.columns:
        df_cluster.drop('_stratify', axis=1, inplace=True)

    # Recreate feature matrix after sampling (encoded columns are already in df_cluster)
    X_cluster = df_cluster[all_features_cluster].copy()
    # Convert boolean to int again
    for col in boolean_features_cluster:
        X_cluster[col] = X_cluster[col].astype(int)
    # Handle missing values
    X_cluster = X_cluster.fillna(0)

# Scale features (after sampling if applicable)
scaler_cluster = StandardScaler()
X_cluster_scaled = scaler_cluster.fit_transform(X_cluster)
X_cluster_sampled = X_cluster_scaled
df_cluster_sampled = df_cluster.copy()

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nData formatting: Encoded {len(categorical_features_cluster)} categorical, scaled {X_cluster_scaled.shape[0]:,} × {X_cluster_scaled.shape[1]} features")

# Visualize data formatting for clustering
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature type breakdown
feat_types = ['Numeric (7)', 'Boolean (8)', 'Encoded Cat (5)']
type_cnts = [7, 8, 5]
axes[0].bar(feat_types, type_cnts, color=['#3498db', '#2ecc71', '#e67e22'], edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title(f'Feature Types (Total: {X_cluster_scaled.shape[1]})', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(type_cnts):
    axes[0].text(i, val, f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Sample size visualization
if len(df_cluster) < len(X_cluster_scaled):
    original_size = len(df_cluster)
else:
    original_size = len(X_cluster_scaled) * 2  # Estimate
axes[1].bar(['Before Sampling', 'After Sampling'], [original_size, X_cluster_scaled.shape[0]], 
            color=['#95a5a6', '#9b59b6'], edgecolor='black', alpha=0.7)
axes[1].set_ylabel('Number of Samples', fontsize=12)
axes[1].set_title('Data Sampling', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate([original_size, X_cluster_scaled.shape[0]]):
    axes[1].text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 5.2. Modelling
# ============================================================================
print_step_header("5.2", "Modelling")

# Store feature names for later use
all_features_cluster_final = all_features_cluster

# ============================================================================
# 5.2.1. Select modeling techniques
# ============================================================================
print_step_header("5.2.1", "Select modeling techniques")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print("\nModel selection: KMeans Clustering (unsupervised, pattern identification, interpretable)")

# ============================================================================
# 5.2.2. Generate test design
# ============================================================================
print_step_header("5.2.2", "Generate test design")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
inertias              = None  # List to store inertia values
silhouette_scores     = None  # List to store silhouette scores
davies_bouldin_scores = None  # List to store Davies-Bouldin scores
k_range               = None  # Range of k values to test
optimal_k             = None  # Selected optimal k

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Select optimal k (you can adjust based on the plots)
optimal_k = 5  # This can be adjusted based on elbow curve

inertias = []
silhouette_scores = []
davies_bouldin_scores = []
k_range = range(2, 11)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=2025, n_init=10, max_iter=300)
    cluster_labels = kmeans.fit_predict(X_cluster_sampled)

    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_cluster_sampled, cluster_labels))
    davies_bouldin_scores.append(davies_bouldin_score(X_cluster_sampled, cluster_labels))

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nTest design: Elbow Method + Silhouette + Davies-Bouldin, testing k={min(k_range)}-{max(k_range)}")

# Plot elbow curve
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Inertia (Elbow Method)
axes[0].plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia (Within-cluster sum of squares)')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True, alpha=0.3)

# Silhouette Score (higher is better)
axes[1].plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score for Different k')
axes[1].grid(True, alpha=0.3)

# Davies-Bouldin Score (lower is better)
axes[2].plot(k_range, davies_bouldin_scores, 'ro-', linewidth=2, markersize=8)
axes[2].set_xlabel('Number of Clusters (k)')
axes[2].set_ylabel('Davies-Bouldin Score')
axes[2].set_title('Davies-Bouldin Score for Different k')
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"\nSelected optimal k: {optimal_k}")

# ============================================================================
# 5.2.3. Build model
# ============================================================================
print_step_header("5.2.3", "Build model")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
kmeans_final   = None  # Final KMeans model
cluster_labels = None  # Cluster labels for each record

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
kmeans_final = KMeans(n_clusters=optimal_k, random_state=2025, n_init=10, max_iter=300)
cluster_labels = kmeans_final.fit_predict(X_cluster_sampled)

df_cluster_sampled['Cluster'] = cluster_labels

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
sil_score = silhouette_score(X_cluster_sampled, cluster_labels)
db_score = davies_bouldin_score(X_cluster_sampled, cluster_labels)
print(f"\nKMeans Clustering: k={optimal_k}, Silhouette={sil_score:.3f}, DB={db_score:.3f}, Inertia={kmeans_final.inertia_:.0f}")

# Visualize cluster distribution
cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
cluster_pcts = (cluster_counts / len(cluster_labels) * 100).round(1)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Bar chart
axes[0].bar(cluster_counts.index.astype(str), cluster_counts.values, 
            color='steelblue', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Cluster ID', fontsize=12)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title('Cluster Distribution (Counts)', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, (idx, val) in enumerate(cluster_counts.items()):
    axes[0].text(i, val, f'{val:,}\n({cluster_pcts[idx]:.1f}%)', 
                 ha='center', va='bottom', fontsize=9)

# Pie chart
axes[1].pie(cluster_counts.values, labels=[f'Cluster {i}' for i in cluster_counts.index],
             autopct='%1.1f%%', startangle=90, colors=plt.cm.viridis(np.linspace(0, 1, len(cluster_counts))))
axes[1].set_title('Cluster Distribution (Percentages)', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 5.2.4. Assess model
# ============================================================================
print_step_header("5.2.4", "Assess model")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
cluster_characteristics = {}  # Store cluster characteristics
X_pca = None  # PCA projection for visualization
pca = None  # PCA model
cluster_outliers = None  # Outlier flags
outlier_threshold = None  # Outlier threshold
n_cluster_outliers = None  # Number of outliers

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Calculate cluster characteristics
for cluster_id in range(optimal_k):
    cluster_data = df_cluster_sampled[df_cluster_sampled['Cluster'] == cluster_id]
    cluster_characteristics[cluster_id] = {
        'size': len(cluster_data),
        'pct': len(cluster_data) / len(df_cluster_sampled) * 100,
        'hour_mode': cluster_data['Hour'].mode().values[0] if len(cluster_data['Hour'].mode()) > 0 else None,
        'month_mode': cluster_data['Month'].mode().values[0] if len(cluster_data['Month'].mode()) > 0 else None,
        'weekend_pct': cluster_data['IsWeekend'].mean() * 100,
        'accident_pct': cluster_data['Accident'].mean() * 100,
        'alcohol_pct': cluster_data['Alcohol'].mean() * 100,
        'vehicle_age_mean': cluster_data['VehicleAge'].mean(),
        'top_violations': cluster_data['Violation Type'].value_counts().head(3).to_dict()
    }

# Calculate PCA for visualization
pca = PCA(n_components=2, random_state=2025)
X_pca = pca.fit_transform(X_cluster_sampled)

# Calculate outliers based on distance from cluster centers
distances = kmeans_final.transform(X_cluster_sampled)
min_distances = distances.min(axis=1)
outlier_threshold = np.percentile(min_distances, 99)
cluster_outliers = min_distances > outlier_threshold
n_cluster_outliers = cluster_outliers.sum()

# Store results in dataframe
df_cluster_sampled['Cluster_Outlier'] = cluster_outliers
df_cluster_sampled['Distance_to_Center'] = min_distances

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nCluster Characteristics: {optimal_k} clusters analyzed with key features")

# Prepare data for visualization
char_data = []
for cluster_id in range(optimal_k):
    char = cluster_characteristics[cluster_id]
    char_data.append({
        'Cluster': cluster_id,
        'Size': char['size'],
        'Weekend%': char['weekend_pct'],
        'Accident%': char['accident_pct'],
        'Alcohol%': char['alcohol_pct'],
        'VehicleAge': char['vehicle_age_mean']
    })
char_df = pd.DataFrame(char_data)

# Visualize cluster characteristics
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Cluster sizes
axes[0, 0].bar(char_df['Cluster'].astype(str), char_df['Size'], 
               color='steelblue', edgecolor='black', alpha=0.7)
axes[0, 0].set_xlabel('Cluster ID', fontsize=11)
axes[0, 0].set_ylabel('Count', fontsize=11)
axes[0, 0].set_title('Cluster Sizes', fontsize=12, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='y')

# Weekend percentage
axes[0, 1].bar(char_df['Cluster'].astype(str), char_df['Weekend%'], 
               color='#2ecc71', edgecolor='black', alpha=0.7)
axes[0, 1].set_xlabel('Cluster ID', fontsize=11)
axes[0, 1].set_ylabel('Weekend %', fontsize=11)
axes[0, 1].set_title('Weekend Percentage by Cluster', fontsize=12, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

# Accident and Alcohol percentages
x = np.arange(len(char_df))
width = 0.35
axes[1, 0].bar(x - width/2, char_df['Accident%'], width, label='Accident%', 
               color='#e74c3c', edgecolor='black', alpha=0.7)
axes[1, 0].bar(x + width/2, char_df['Alcohol%'], width, label='Alcohol%', 
               color='#f39c12', edgecolor='black', alpha=0.7)
axes[1, 0].set_xlabel('Cluster ID', fontsize=11)
axes[1, 0].set_ylabel('Percentage', fontsize=11)
axes[1, 0].set_title('Accident & Alcohol Percentages', fontsize=12, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(char_df['Cluster'].astype(str))
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Vehicle age
axes[1, 1].bar(char_df['Cluster'].astype(str), char_df['VehicleAge'], 
               color='#9b59b6', edgecolor='black', alpha=0.7)
axes[1, 1].set_xlabel('Cluster ID', fontsize=11)
axes[1, 1].set_ylabel('Average Vehicle Age (years)', fontsize=11)
axes[1, 1].set_title('Average Vehicle Age by Cluster', fontsize=12, fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# Visualizations
# Geographic distribution
plt.figure(figsize=(12, 8))
plt.scatter(df_cluster_sampled['Longitude'], df_cluster_sampled['Latitude'],
            c=df_cluster_sampled['Cluster'], cmap='viridis', alpha=0.6, s=10)
plt.colorbar(label='Cluster')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.title(f'KMeans Clustering (k={optimal_k}) - Geographic Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# PCA projection
plt.figure(figsize=(12, 8))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=cluster_labels, cmap='viridis', alpha=0.6, s=10)
plt.colorbar(label='Cluster')
plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)')
plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)')
plt.title(f'KMeans Clustering (k={optimal_k}) - PCA Projection')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Summary (reuse previously calculated scores)
print(f"\nModel Assessment: k={optimal_k}, Silhouette={sil_score:.3f}, DB={db_score:.3f}, Outliers={n_cluster_outliers:,} ({n_cluster_outliers/len(df_cluster_sampled)*100:.2f}%)")

# ============================================================================
# 5.3. Evaluation
# ============================================================================
print_step_header("5.3", "Evaluation")

# ============================================================================
# 5.3.1. Evaluate results
# ============================================================================
print_step_header("5.3.1", "Evaluate results")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nClustering Results: {len(df_cluster_sampled):,} records, k={optimal_k}, Silhouette={sil_score:.3f}, DB={db_score:.3f}, Inertia={kmeans_final.inertia_:.0f}")

# Visualize clustering metrics
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Silhouette Score
axes[0].bar(['Silhouette'], [sil_score], color='#3498db', edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Score', fontsize=11)
axes[0].set_title(f'Silhouette Score\n({sil_score:.3f})', fontsize=12, fontweight='bold')
axes[0].set_ylim([0, 1])
axes[0].grid(True, alpha=0.3, axis='y')
axes[0].text(0, sil_score + 0.05, f'{sil_score:.3f}', ha='center', fontsize=10, fontweight='bold')

# Davies-Bouldin Score (lower is better)
axes[1].bar(['DB Score'], [db_score], color='#e74c3c', edgecolor='black', alpha=0.7)
axes[1].set_ylabel('Score', fontsize=11)
axes[1].set_title(f'Davies-Bouldin Score\n({db_score:.3f}, lower is better)', fontsize=12, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
axes[1].text(0, db_score + db_score*0.1, f'{db_score:.3f}', ha='center', fontsize=10, fontweight='bold')

# Inertia
axes[2].bar(['Inertia'], [kmeans_final.inertia_], color='#2ecc71', edgecolor='black', alpha=0.7)
axes[2].set_ylabel('Inertia', fontsize=11)
axes[2].set_title(f'Inertia\n({kmeans_final.inertia_:.0f})', fontsize=12, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')
axes[2].text(0, kmeans_final.inertia_ + kmeans_final.inertia_*0.05, 
             f'{kmeans_final.inertia_:.0f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 5.3.2. Interpret results
# ============================================================================
print_step_header("5.3.2", "Interpret results")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nCluster interpretation: {optimal_k} distinct patterns identified")

# Visualize cluster sizes
fig, ax = plt.subplots(figsize=(10, 6))
cluster_sizes = [len(df_cluster_sampled[df_cluster_sampled['Cluster'] == i]) for i in range(optimal_k)]
cluster_pcts = [size/len(df_cluster_sampled)*100 for size in cluster_sizes]
ax.bar([f'Cluster {i}' for i in range(optimal_k)], cluster_sizes, color=plt.cm.viridis(np.linspace(0, 1, optimal_k)), edgecolor='black', alpha=0.7)
ax.set_ylabel('Number of Records', fontsize=12)
ax.set_title(f'Cluster Size Distribution (k={optimal_k})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, (size, pct) in enumerate(zip(cluster_sizes, cluster_pcts)):
    ax.text(i, size, f'{size:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 5.3.3. Review of process
# ============================================================================
print_step_header("5.3.3", "Review of process")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print("\nProcess review: Complete clustering workflow from data preparation to evaluation")

# Visualize clustering process workflow
fig, ax = plt.subplots(figsize=(12, 6))
steps_cluster = ['Data\nSelection', 'Data\nCleaning', 'Feature\nEngineering', 
                 'Feature\nIntegration', 'Data\nFormatting', 'k\nOptimization', 'Clustering', 'Assessment']
status_cluster = [1, 1, 1, 1, 1, 1, 1, 1]  # All completed
colors_proc = ['#2ecc71' if s == 1 else '#95a5a6' for s in status_cluster]

y_pos_cluster = np.arange(len(steps_cluster))
ax.barh(y_pos_cluster, status_cluster, color=colors_proc, edgecolor='black', alpha=0.7)
ax.set_yticks(y_pos_cluster)
ax.set_yticklabels(steps_cluster, fontsize=11)
ax.set_xlabel('Status', fontsize=12)
ax.set_title('Clustering Process Review', fontsize=14, fontweight='bold')
ax.set_xlim([0, 1.2])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not Started', 'Completed'])
ax.invert_yaxis()

for i, (step, s) in enumerate(zip(steps_cluster, status_cluster)):
    ax.text(1.05, i, '✓ Completed', va='center', fontsize=10, fontweight='bold', color='#2ecc71')

plt.tight_layout()
plt.show()

# ============================================================================
# 5.3.4. Determine next steps
# ============================================================================
print_step_header("5.3.4", "Determine next steps")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nClustering complete: k={optimal_k}, {len(df_cluster_sampled):,} records, Silhouette={sil_score:.3f}")

# Visualize clustering completion summary
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Clustering metrics
metric_names = ['Silhouette', 'DB Score', 'Inertia/1000']
metric_vals = [sil_score, db_score, kmeans_final.inertia_/1000]
metric_colors = ['#3498db', '#e74c3c', '#2ecc71']
axes[0].bar(metric_names, metric_vals, color=metric_colors, edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Clustering Quality Metrics', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(metric_vals):
    axes[0].text(i, val, f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Dataset summary
data_info = ['Records', 'Features', 'Clusters']
data_vals = [len(df_cluster_sampled), len(all_features_cluster), optimal_k]
axes[1].bar(data_info, data_vals, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Dataset Summary', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(data_vals):
    axes[1].text(i, val, f'{val:,}' if val > 100 else f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Cluster distribution
cluster_szs = [len(df_cluster_sampled[df_cluster_sampled['Cluster'] == i]) for i in range(optimal_k)]
axes[2].bar([f'C{i}' for i in range(optimal_k)], cluster_szs, color=plt.cm.viridis(np.linspace(0, 1, optimal_k)), edgecolor='black', alpha=0.7)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('Cluster Sizes', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================================================
# 6. Outlier Detection by LOF and Distance-based method (and common outliers)
# ============================================================================
print_step_header("6", "Outlier Detection by LOF and Distance-based method")

# ============================================================================
# 6.1. Data Preparation for Outlier Detection
# ============================================================================
print_step_header("6.1", "Data Preparation for Outlier Detection")

df_outlier = df.copy()

# ============================================================================
# 6.1.1. Select Data
# ============================================================================

print_step_header("6.1.1", "Select Data - Select and justify features for outlier detection")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
feature_groups         = None  # Dictionary of feature groups
geographic_features    = None  # Geographic feature list
temporal_features      = None  # Temporal feature list
severity_features      = None  # Severity feature list
risk_features          = None  # Risk feature list
vehicle_features       = None  # Vehicle feature list
safety_features        = None  # Safety feature list
demographic_features   = None  # Demographic feature list
enforcement_features   = None  # Enforcement feature list
auxiliary_features     = None  # Auxiliary feature list
selected_features      = None  # Consolidated list of all selected features
modeling_features_count = None  # Count of modeling features (excluding auxiliary)

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Define feature groups (10+ modeling attributes required)
feature_groups = {
    'Geographic':   ['Latitude', 'Longitude'],
    'Temporal':     ['Date Of Stop', 'Time Of Stop'],
    'Severity':     ['Accident', 'Personal Injury', 'Property Damage', 'Fatal'],  # KEY
    'Risk':         ['Alcohol', 'Work Zone'],
    'Vehicle':      ['Year', 'VehicleType', 'Commercial Vehicle', 'HAZMAT'],
    'Safety':       ['Belts', 'Contributed To Accident'],
    'Demographics': ['Gender', 'Race'],
    'Enforcement':  ['SubAgency'],
    'Auxiliary':    ['SeqID', 'Location', 'Violation Type']  # Non-modeling
}

# Extract feature lists
geographic_features  = feature_groups['Geographic']
temporal_features    = feature_groups['Temporal']
severity_features    = feature_groups['Severity']
risk_features        = feature_groups['Risk']
vehicle_features     = feature_groups['Vehicle']
safety_features      = feature_groups['Safety']
demographic_features = feature_groups['Demographics']
enforcement_features = feature_groups['Enforcement']
auxiliary_features   = feature_groups['Auxiliary']

# Consolidate all features
selected_features = (
    auxiliary_features + geographic_features + temporal_features +
    severity_features + risk_features + vehicle_features +
    safety_features + demographic_features + enforcement_features
)

# Filter dataset to selected features
df_outlier = df_outlier[selected_features].copy()

modeling_features_count = len(selected_features) - len(auxiliary_features)

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nOutlier detection setup: {df_outlier.shape[0]:,} rows × {df_outlier.shape[1]} columns, {modeling_features_count} modeling features")

# Visualize feature selection for outlier detection
fig, ax = plt.subplots(figsize=(10, 6))
feature_categories_od = ['Geographic', 'Temporal', 'Severity', 'Risk', 'Vehicle', 'Safety', 'Demographics', 'Enforcement']
feature_counts_od = [len(feature_groups['Geographic']), len(feature_groups['Temporal']), 
                     len(feature_groups['Severity']), len(feature_groups['Risk']), 
                     len(feature_groups['Vehicle']), len(feature_groups['Safety']), 
                     len(feature_groups['Demographics']), len(feature_groups['Enforcement'])]
ax.barh(feature_categories_od, feature_counts_od, color='#e74c3c', edgecolor='black', alpha=0.7)
ax.set_xlabel('Number of Features', fontsize=12)
ax.set_title(f'Feature Selection for Outlier Detection (Total: {modeling_features_count})', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
ax.invert_yaxis()
for i, val in enumerate(feature_counts_od):
    ax.text(val, i, f' {val}', va='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 6.1.2. Clean Data
# ============================================================================

print_step_header("6.1.2", "Clean Data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
rows_before        = None  # Number of rows before deduplication
rows_after         = None  # Number of rows after deduplication
duplicates_removed = None  # Number of duplicates removed
missing_stats      = None  # Missing value counts
missing_stats_pct  = None  # Missing value percentages
missing_cols       = None  # Columns with missing values
invalid_lat        = None  # Count of invalid latitude values
invalid_lon        = None  # Count of invalid longitude values
invalid_year       = None  # Count of invalid year values
invalid_date       = None  # Count of invalid date values
invalid_date_range = None  # Count of dates out of range
invalid_time       = None  # Count of invalid time values
binary_fields      = None  # List of binary fields
invalid_binary     = None  # Count of invalid binary values
rows_before_val    = None  # Rows before validation
valid_mask         = None  # Boolean mask for valid rows
rows_removed       = None  # Number of rows removed
total_invalid      = None  # Total count of invalid values
original_size      = None  # Original dataset size
target_sample_size = None  # Target sample size
sample_fraction    = None  # Fraction to sample
accident_dist      = None  # Distribution of accident class

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# 1. Remove duplicate rows
rows_before = df_outlier.shape[0]
df_outlier.drop_duplicates(subset=['SeqID'], keep='first', inplace=True)
rows_after = df_outlier.shape[0]
duplicates_removed = rows_before - rows_after

# 2. Check missing values
missing_stats = df_outlier.isnull().sum()
missing_stats_pct = (missing_stats / len(df_outlier) * 100).round(2)
missing_cols = missing_stats[missing_stats > 0]

# 3. Data validation and remove invalid records
# Geographic coordinate validation
invalid_lat = ((df_outlier['Latitude'] < -90) | (df_outlier['Latitude'] > 90)).sum()
invalid_lon = ((df_outlier['Longitude'] < -180) | (df_outlier['Longitude'] > 180)).sum()

# Temporal validation
invalid_year = ((df_outlier['Year'] < 1900) | (df_outlier['Year'] > 2025)).sum()
invalid_date = df_outlier['Date Of Stop'].isnull().sum()
invalid_date_range = ((df_outlier['Date Of Stop'] < '2015-01-01') | (df_outlier['Date Of Stop'] > '2025-12-31')).sum()

# Time validation
df_outlier['Time Of Stop Parsed'] = pd.to_datetime(df_outlier['Time Of Stop'], format='%H:%M:%S', errors='coerce')
invalid_time = df_outlier['Time Of Stop Parsed'].isnull().sum()

# Binary fields validation
binary_fields = [
    'Accident', 'Personal Injury', 'Property Damage', 'Fatal',
    'Alcohol', 'Work Zone', 'Belts', 'Contributed To Accident',
    'Commercial Vehicle', 'HAZMAT'
]
invalid_binary = sum((~df_outlier[field].isin([True, False])).sum() for field in binary_fields)

# Remove invalid records
rows_before_val = df_outlier.shape[0]
valid_mask = (
    (df_outlier['Latitude'] >= -90) & (df_outlier['Latitude'] <= 90) &
    (df_outlier['Longitude'] >= -180) & (df_outlier['Longitude'] <= 180) &
    (df_outlier['Year'] >= 1900) & (df_outlier['Year'] <= 2025) &
    (df_outlier['Date Of Stop'].notnull()) &
    (df_outlier['Date Of Stop'] >= '2015-01-01') & (df_outlier['Date Of Stop'] <= '2025-12-31') &
    (df_outlier['Time Of Stop Parsed'].notnull())
)
for field in binary_fields:
    valid_mask &= df_outlier[field].isin([True, False])

df_outlier = df_outlier[valid_mask]
df_outlier.drop('Time Of Stop Parsed', axis=1, inplace=True)
rows_removed = rows_before_val - df_outlier.shape[0]
total_invalid = invalid_lat + invalid_lon + invalid_year + invalid_date + invalid_date_range + invalid_time + invalid_binary

# 4. Apply stratified sampling
original_size = len(df_outlier)
target_sample_size = 10000

if original_size > target_sample_size:
    sample_fraction = target_sample_size / original_size
    df_outlier = df_outlier.groupby('Accident', group_keys=False).apply(
        lambda x: x.sample(frac=sample_fraction, random_state=2025)
    )
    accident_dist = df_outlier['Accident'].value_counts(normalize=True).round(4)

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
dup_info = f", removed {duplicates_removed:,} duplicates" if duplicates_removed > 0 else ", no duplicates"
missing_info = f", {len(missing_cols)} cols with missing values" if not missing_cols.empty else ", no missing values"
invalid_info = f", removed {rows_removed:,} invalid records" if total_invalid > 0 else ""
sampling_info = f", sampled {len(df_outlier):,} ({len(df_outlier)/original_size*100:.1f}%)" if original_size > target_sample_size else f", using all {original_size:,}"
print(f"\nData cleaning: {rows_before:,} → {df_outlier.shape[0]:,} rows (removed missing coordinates)")

# Visualize data cleaning for outlier detection
fig, ax = plt.subplots(figsize=(8, 5))
ax.bar(['Before', 'After'], [rows_before, df_outlier.shape[0]], 
       color=['#95a5a6', '#2ecc71'], edgecolor='black', alpha=0.7)
ax.set_ylabel('Number of Records', fontsize=12)
ax.set_title('Outlier Detection Data Cleaning', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, val in enumerate([rows_before, df_outlier.shape[0]]):
    pct = val / rows_before * 100
    ax.text(i, val, f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 6.1.3. Construct Data
# ============================================================================

print_step_header("6.1.3", "Construct Data - Feature Engineering")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
current_year      = None  # Current year for age calculation
rows_before       = None  # Rows before missing value handling
missing_stats     = None  # Missing value counts
missing_stats_pct = None  # Missing value percentages
rows_after        = None  # Rows after missing value handling
removed           = None  # Number of records removed

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# 1. Create temporal features
# Combine date and time
df_outlier['DateTime'] = pd.to_datetime(
    df_outlier['Date Of Stop'].astype(str) + ' ' + df_outlier['Time Of Stop'].astype(str),
    errors='coerce'
)

# Extract temporal components
df_outlier['Hour'] = df_outlier['DateTime'].dt.hour
df_outlier['Month'] = df_outlier['DateTime'].dt.month
df_outlier['DayOfWeek'] = df_outlier['DateTime'].dt.dayofweek  # 0=Monday, 6=Sunday
df_outlier['IsWeekend'] = (df_outlier['DayOfWeek'] >= 5).astype(int)
df_outlier['TimeOfDay'] = df_outlier['Hour'].apply(categorize_time_of_day)

# 2. Create vehicle age feature
current_year = 2025
df_outlier['VehicleAge'] = current_year - df_outlier['Year']
# Handle outliers: cap at reasonable maximum (50 years for very old vehicles)
df_outlier['VehicleAge'] = df_outlier['VehicleAge'].clip(lower=0, upper=50)

# 3. Create binning features (optional)
# VehicleAge binning using if conditions
def bin_vehicle_age(age):
    if age <= 3:
        return 'New'
    elif age <= 7:
        return 'Recent'
    elif age <= 15:
        return 'Middle'
    else:
        return 'Old'

df_outlier['VehicleAge_Binned'] = df_outlier['VehicleAge'].apply(bin_vehicle_age)

# Hour binning using if conditions
def bin_hour(hour):
    if hour <= 6:
        return 'Night'
    elif hour <= 12:
        return 'Morning'
    elif hour <= 18:
        return 'Afternoon'
    else:
        return 'Evening'

df_outlier['Hour_Binned'] = df_outlier['Hour'].apply(bin_hour)

# 4. Handle missing values after feature engineering
rows_before = df_outlier.shape[0]
missing_stats = df_outlier.isnull().sum()
missing_stats_pct = (missing_stats / len(df_outlier) * 100).round(2)

# Strategy: Remove rows with missing values
if missing_stats[missing_stats > 0].any():
    df_outlier.dropna(inplace=True)
    rows_after = df_outlier.shape[0]
else:
    rows_after = rows_before

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
missing_info = f", removed {rows_before - rows_after:,} records with missing values" if missing_stats[missing_stats > 0].any() else ", no missing values"
print(f"\nFeature engineering: Created temporal features (Hour, Month, DayOfWeek, IsWeekend, TimeOfDay), VehicleAge (mean={df_outlier['VehicleAge'].mean():.1f}yr), binning features{missing_info}")

# Visualize engineered features for outlier detection
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
axes = axes.flatten()

# Hour distribution
df_outlier['Hour'].hist(bins=24, ax=axes[0], edgecolor='black', color='#e74c3c', alpha=0.7)
axes[0].set_title('Hour Distribution', fontsize=11, fontweight='bold')
axes[0].set_xlabel('Hour')
axes[0].set_ylabel('Frequency')
axes[0].grid(True, alpha=0.3)

# Month distribution
df_outlier['Month'].hist(bins=12, ax=axes[1], edgecolor='black', color='#3498db', alpha=0.7)
axes[1].set_title('Month Distribution', fontsize=11, fontweight='bold')
axes[1].set_xlabel('Month')
axes[1].set_ylabel('Frequency')
axes[1].grid(True, alpha=0.3)

# DayOfWeek distribution
df_outlier['DayOfWeek'].hist(bins=7, ax=axes[2], edgecolor='black', color='#2ecc71', alpha=0.7)
axes[2].set_title('Day of Week Distribution', fontsize=11, fontweight='bold')
axes[2].set_xlabel('Day (0=Mon, 6=Sun)')
axes[2].set_ylabel('Frequency')
axes[2].grid(True, alpha=0.3)

# IsWeekend distribution
weekend_cnts_od = df_outlier['IsWeekend'].value_counts()
axes[3].bar(['Weekday', 'Weekend'], [weekend_cnts_od.get(0, 0), weekend_cnts_od.get(1, 0)], 
            color=['#3498db', '#e67e22'], edgecolor='black', alpha=0.7)
axes[3].set_title('Weekend vs Weekday', fontsize=11, fontweight='bold')
axes[3].set_ylabel('Frequency')
axes[3].grid(True, alpha=0.3, axis='y')

# TimeOfDay distribution
time_cnts_od = df_outlier['TimeOfDay'].value_counts()
axes[4].bar(time_cnts_od.index, time_cnts_od.values, color='#9b59b6', edgecolor='black', alpha=0.7)
axes[4].set_title('Time of Day Distribution', fontsize=11, fontweight='bold')
axes[4].set_ylabel('Frequency')
axes[4].tick_params(axis='x', rotation=45)
axes[4].grid(True, alpha=0.3, axis='y')

# VehicleAge distribution
df_outlier['VehicleAge'].hist(bins=30, ax=axes[5], edgecolor='black', color='#1abc9c', alpha=0.7)
axes[5].set_title('Vehicle Age Distribution', fontsize=11, fontweight='bold')
axes[5].set_xlabel('Age (years)')
axes[5].set_ylabel('Frequency')
axes[5].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ============================================================================
# 6.1.4. Integrate Data
# ============================================================================

print_step_header("6.1.4", "Integrate Data")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
numerical_features       = None  # List of numerical features
boolean_features         = None  # List of boolean features
categorical_features     = None  # List of categorical features
optional_binned_features = None  # List of optional binned features
total_features           = None  # Total number of features
arm_features             = None  # Features for Association Rule Mining
arm_data                 = None  # Data for Association Rule Mining
cooccurrence             = None  # Dictionary for co-occurrence patterns
association_rules_df     = None  # Association rules dataframe
correlation_matrix       = None  # Correlation matrix of features
high_corr_pairs          = None  # List of highly correlated feature pairs

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# 1. Define feature categories
numerical_features = ['Latitude', 'Longitude', 'Hour', 'Month', 'DayOfWeek', 'IsWeekend', 'VehicleAge']

boolean_features = [
    'Accident', 'Personal Injury', 'Property Damage', 'Fatal',
    'Alcohol', 'Work Zone', 'Commercial Vehicle', 'HAZMAT',
    'Belts', 'Contributed To Accident'
]

categorical_features = ['VehicleType', 'SubAgency', 'Gender', 'Race', 'TimeOfDay']

# Optional binned features (not included in main modeling)
optional_binned_features = ['VehicleAge_Binned', 'Hour_Binned']

total_features = len(numerical_features) + len(boolean_features) + len(categorical_features)

# 2. Association Rule Mining (Simplified)
# Select key categorical and boolean features for association analysis
arm_features = ['Accident', 'Personal Injury', 'Property Damage', 'Fatal',
               'Alcohol', 'Work Zone', 'Belts', 'Commercial Vehicle', 'HAZMAT']

# Create binary matrix for association rule mining
arm_data = df_outlier[arm_features].copy()
# Convert boolean to int (0/1)
for col in arm_features:
    arm_data[col] = arm_data[col].astype(int)

try:
    # Calculate co-occurrence patterns
    cooccurrence = {}
    for i, feat1 in enumerate(arm_features):
        for feat2 in arm_features[i+1:]:
            both_true = ((arm_data[feat1] == 1) & (arm_data[feat2] == 1)).sum()
            total = len(arm_data)
            if both_true > 0:
                support = both_true / total
                confidence_feat1_to_feat2 = both_true / (arm_data[feat1] == 1).sum() if (arm_data[feat1] == 1).sum() > 0 else 0
                cooccurrence[(feat1, feat2)] = {
                    'support': support,
                    'confidence': confidence_feat1_to_feat2,
                    'count': both_true
                }

    association_rules_df = pd.DataFrame([
        {
            'antecedents': feat1,
            'consequents': feat2,
            'support': metrics['support'],
            'confidence': metrics['confidence'],
            'count': metrics['count']
        }
        for (feat1, feat2), metrics in cooccurrence.items()
    ])

except Exception:
    association_rules_df = None

# 3. Correlation Analysis
# Calculate correlation matrix for numerical features
correlation_matrix = df_outlier[numerical_features].corr()

# Identify highly correlated pairs (|correlation| > 0.8)
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        corr_value = correlation_matrix.iloc[i, j]
        if abs(corr_value) > 0.8:
            high_corr_pairs.append((
                correlation_matrix.columns[i],
                correlation_matrix.columns[j],
                corr_value
            ))

# Visualize correlation matrix
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
assoc_info = f", {len(association_rules_df)} association rules" if association_rules_df is not None and len(association_rules_df) > 0 else ", no association rules"
corr_info = f", {len(high_corr_pairs)} highly correlated pairs" if high_corr_pairs else ", no highly correlated pairs"
print(f"\nFeature integration: {total_features} final features ({len(numerical_features)} numeric, {len(boolean_features)} boolean, {len(categorical_features)} categorical){assoc_info}{corr_info}")

# ============================================================================
# 6.1.5. Format Data
# ============================================================================
print_step_header("6.1.5", "Format Data - Encode, scale, and validate")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
encoders                     = {}    # Dictionary to store label encoders
encoded_categorical_features = None  # List of encoded categorical feature names
all_modeling_features        = None  # List of all features used for modeling
X_outlier                    = None  # Feature matrix for outlier detection
scaler                       = None  # StandardScaler object
X_outlier_scaled             = None  # Scaled feature matrix (numpy array)
X_outlier_scaled_df          = None  # Scaled feature matrix (DataFrame)
nan_count                    = None  # Count of NaN values in scaled data
inf_count                    = None  # Count of infinite values in scaled data

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# 1. Encode categorical variables
for feature in categorical_features:
    # Apply LabelEncoder
    le = LabelEncoder()
    # Fill NaN with 'Unknown' before encoding
    df_outlier[feature] = df_outlier[feature].fillna('Unknown')
    df_outlier[f'{feature}_encoded'] = le.fit_transform(df_outlier[feature].astype(str))

    # Save encoder for later interpretation
    encoders[feature] = le

# 2. Create feature matrix and validate data types
# Create feature matrix: integrate numerical, boolean, encoded features
encoded_categorical_features = [f'{f}_encoded' for f in categorical_features]
all_modeling_features = numerical_features + boolean_features + encoded_categorical_features

# Convert boolean features to integers (0/1)
for feature in boolean_features:
    df_outlier[feature] = df_outlier[feature].astype(int)

X_outlier = df_outlier[all_modeling_features].copy()

# 3. Feature scaling
# Initialize scaler
scaler = StandardScaler()

# Fit and transform features
X_outlier_scaled = scaler.fit_transform(X_outlier)

# Convert back to DataFrame for readability
X_outlier_scaled_df = pd.DataFrame(
    X_outlier_scaled,
    columns=X_outlier.columns,
    index=X_outlier.index
)

# 4. Final validation
nan_count = np.isnan(X_outlier_scaled).sum()
inf_count = np.isinf(X_outlier_scaled).sum()

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
quality_info = f", found NaN: {nan_count}, Inf: {inf_count}" if nan_count > 0 or inf_count > 0 else ", quality check passed"
print(f"\nData formatting: Encoded {len(categorical_features)} categorical, scaled {X_outlier_scaled.shape[0]:,} rows × {X_outlier_scaled.shape[1]} columns{quality_info}")

# Visualize data formatting for outlier detection
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Feature type breakdown
feat_types_od = ['Numeric (7)', 'Boolean (10)', 'Encoded Cat (5)'] # Updated boolean count
type_cnts_od = [len(numerical_features), len(boolean_features), len(categorical_features)]
axes[0].bar(feat_types_od, type_cnts_od, color=['#3498db', '#2ecc71', '#e67e22'], edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Count', fontsize=12)
axes[0].set_title(f'Feature Types (Total: {X_outlier_scaled.shape[1]})', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(type_cnts_od):
    axes[0].text(i, val, f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Sample size
axes[1].bar(['Original', 'After Sampling'], [original_size, X_outlier_scaled.shape[0]], 
            color=['#95a5a6', '#e74c3c'], edgecolor='black', alpha=0.7)
axes[1].set_ylabel('Number of Samples', fontsize=12)
axes[1].set_title('Sampling for Outlier Detection', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate([original_size, X_outlier_scaled.shape[0]]):
    axes[1].text(i, val, f'{val:,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 6.2. Modelling
# ============================================================================
print_step_header("6.2", "Modelling")

# Store feature names for later use
all_features = all_modeling_features

# ============================================================================
# 6.2.1. Select modeling techniques
# ============================================================================
print_step_header("6.2.1", "Select modeling techniques")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print("\nModel selection: LOF + Distance-based Outlier Detection (as per project requirements)")

# ============================================================================
# 6.2.2. Generate test design
# ============================================================================
print_step_header("6.2.2", "Generate test design")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print("\nTest design: LOF (n_neighbors=20, contamination=0.01) + Distance-based (k=20, 99th percentile) → Common outliers for high confidence")

# ============================================================================
# 6.2.3. Build model
# ============================================================================
print_step_header("6.2.3", "Build model")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
lof                  = None  # Local Outlier Factor model
lof_labels           = None  # Labels predicted by LOF
outliers_lof         = None  # Boolean mask for LOF outliers
lof_scores           = None  # LOF anomaly scores
distance_outliers    = None  # Boolean mask for distance-based outliers
distance_scores      = None  # Distance-based anomaly scores
distance_threshold   = None  # Threshold for distance-based outliers
k_neighbors          = None  # Number of neighbors for distance calculation
nn                   = None  # NearestNeighbors object
distances            = None  # Distance matrix
indices              = None  # Indices of nearest neighbors
avg_distances        = None  # Average distances to k nearest neighbors

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# 1. Build LOF Model
n_neighbors_lof = 20 # Parameter for LOF
lof = LocalOutlierFactor(n_neighbors=n_neighbors_lof, contamination=0.01, novelty=False)
lof_labels = lof.fit_predict(X_outlier_scaled)

# -1 for outliers, 1 for inliers
outliers_lof = (lof_labels == -1)

# Get LOF scores (negative outlier factor)
lof_scores = lof.negative_outlier_factor_
df_outlier['LOF_score'] = lof_scores
df_outlier['LOF_outlier'] = outliers_lof

# 2. Build Distance-based Outlier Detection Model
# Method: Calculate average distance to k nearest neighbors
# Points with high average distance are considered outliers
k_neighbors_dist = 20  # Number of neighbors to consider
nn = NearestNeighbors(n_neighbors=k_neighbors_dist + 1)  # +1 because point itself is included
nn.fit(X_outlier_scaled)
distances, indices = nn.kneighbors(X_outlier_scaled)

# Calculate average distance to k nearest neighbors (excluding the point itself)
# The first column is the point itself (distance = 0), so we take columns 1 to k+1
avg_distances = distances[:, 1:].mean(axis=1)

# Use percentile-based threshold (top 1% as outliers, matching contamination=0.01)
distance_threshold = np.percentile(avg_distances, 99)
distance_outliers = avg_distances > distance_threshold

# Store results
distance_scores = avg_distances
df_outlier['Distance_score'] = distance_scores
df_outlier['Distance_outlier'] = distance_outliers

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nModel building: LOF and Distance-based models trained on {len(X_outlier_scaled):,} samples")

# Visualize model building summary
fig, ax = plt.subplots(figsize=(8, 5))
models = ['LOF', 'Distance-based']
model_params = [n_neighbors_lof, k_neighbors_dist]
ax.bar(models, model_params, color=['#e74c3c', '#3498db'], edgecolor='black', alpha=0.7)
ax.set_ylabel('k/n_neighbors Parameter', fontsize=12)
ax.set_title('Outlier Detection Models Configuration', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, val in enumerate(model_params):
    ax.text(i, val, f'k={val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 6.2.4. Assess model
# ============================================================================
print_step_header("6.2.4", "Assess model")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------
n_outliers_lof         = None  # Number of outliers detected by LOF
n_outliers_distance    = None  # Number of outliers detected by distance method
common_outliers        = None  # Boolean mask for common outliers
n_common_outliers      = None  # Number of common outliers
common_outlier_df      = None  # DataFrame of common outliers

# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------
# Count outliers
n_outliers_lof = outliers_lof.sum()
n_outliers_distance = distance_outliers.sum()

# Find common outliers (detected by both methods)
common_outliers = outliers_lof & distance_outliers
df_outlier['Common_outlier'] = common_outliers
n_common_outliers = common_outliers.sum()

# Analyze common outliers if any
if n_common_outliers > 0:
    common_outlier_df = df_outlier[common_outliers]

# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
lof_pct = n_outliers_lof/len(df_outlier)*100
distance_pct = n_outliers_distance/len(df_outlier)*100
common_pct = n_common_outliers/len(df_outlier)*100
print(f"\nOutlier Detection: LOF {n_outliers_lof:,} ({lof_pct:.2f}%), Distance {n_outliers_distance:,} ({distance_pct:.2f}%), Common {n_common_outliers:,} ({common_pct:.3f}%)")

# Visualization: comparison chart
lof_only = n_outliers_lof - n_common_outliers
distance_only = n_outliers_distance - n_common_outliers
fig, ax = plt.subplots(figsize=(8, 5))
categories = ['LOF Only', 'Distance Only', 'Both Methods']
counts = [lof_only, distance_only, n_common_outliers]
ax.bar(categories, counts, color=['#ff7f7f', '#ffb347', '#9370db'], edgecolor='black', alpha=0.7)
ax.set_ylabel('Number of Outliers', fontsize=12)
ax.set_title('Outlier Detection Method Comparison', fontsize=13, fontweight='bold')
for i, (cat, count) in enumerate(zip(categories, counts)):
    ax.text(i, count, f'{count:,}\n({count/len(df_outlier)*100:.2f}%)',
            ha='center', va='bottom', fontsize=10)
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.show()

if n_common_outliers > 0:
    lof_overlap = n_common_outliers / n_outliers_lof * 100 if n_outliers_lof > 0 else 0
    distance_overlap = n_common_outliers / n_outliers_distance * 100 if n_outliers_distance > 0 else 0
    print(f"Method Agreement: {lof_overlap:.1f}% of LOF outliers, {distance_overlap:.1f}% of Distance-based outliers")

# ============================================================================
# 6.3. Evaluation
# ============================================================================
print_step_header("6.3", "Evaluation")

# ============================================================================
# 6.3.1. Evaluate results
# ============================================================================
print_step_header("6.3.1", "Evaluate results")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nEvaluation Summary: {len(df_outlier):,} records, LOF {n_outliers_lof:,} ({n_outliers_lof/len(df_outlier)*100:.2f}%), Distance {n_outliers_distance:,} ({n_outliers_distance/len(df_outlier)*100:.2f}%), Common {n_common_outliers:,} ({n_common_outliers/len(df_outlier)*100:.3f}%)")

# Visualization: comparison chart
lof_only = n_outliers_lof - n_common_outliers
distance_only = n_outliers_distance - n_common_outliers
fig, axes = plt.subplots(1, 2, figsize=(16, 5))

# Method comparison
categories = ['LOF Only', 'Distance Only', 'Both Methods']
counts = [lof_only, distance_only, n_common_outliers]
axes[0].bar(categories, counts, color=['#ff7f7f', '#ffb347', '#9370db'], edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Number of Outliers', fontsize=12)
axes[0].set_title('Outlier Detection Method Comparison', fontsize=13, fontweight='bold')
for i, (cat, count) in enumerate(zip(categories, counts)):
    axes[0].text(i, count, f'{count:,}\n({count/len(df_outlier)*100:.2f}%)',
            ha='center', va='bottom', fontsize=10)
axes[0].grid(True, alpha=0.3, axis='y')

# Outlier characteristics comparison (if common outliers exist)
if n_common_outliers > 0:
    normal_df = df_outlier[~common_outliers]
    violation_type_counts = common_outlier_df['Violation Type'].value_counts()
    
    # Prepare comparison data
    comparison_data = {
        'Accident Rate (%)': [common_outlier_df['Accident'].mean()*100, normal_df['Accident'].mean()*100],
        'Alcohol %': [common_outlier_df['Alcohol'].mean()*100, normal_df['Alcohol'].mean()*100],
        'Vehicle Age (yr)': [common_outlier_df['VehicleAge'].mean(), normal_df['VehicleAge'].mean()]
    }
    
    x = np.arange(len(comparison_data))
    width = 0.35
    for i, (metric, values) in enumerate(comparison_data.items()):
        axes[1].bar(x[i] - width/2, values[0], width, label='Outliers' if i == 0 else '', 
                   color='#e74c3c', edgecolor='black', alpha=0.7)
        axes[1].bar(x[i] + width/2, values[1], width, label='Normal' if i == 0 else '', 
                   color='#3498db', edgecolor='black', alpha=0.7)
    
    axes[1].set_ylabel('Value', fontsize=12)
    axes[1].set_title('Outlier vs Normal Characteristics', fontsize=13, fontweight='bold')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(list(comparison_data.keys()), rotation=15, ha='right')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for i, (metric, values) in enumerate(comparison_data.items()):
        axes[1].text(i - width/2, values[0], f'{values[0]:.1f}', ha='center', va='bottom', fontsize=9)
        axes[1].text(i + width/2, values[1], f'{values[1]:.1f}', ha='center', va='bottom', fontsize=9)
else:
    axes[1].text(0.5, 0.5, 'No common outliers\ndetected', ha='center', va='center', 
                transform=axes[1].transAxes, fontsize=12)
    axes[1].set_title('Outlier Characteristics', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# 6.3.2. Interpret results
# ============================================================================
print_step_header("6.3.2", "Interpret results")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nOutlier interpretation: {n_common_outliers:,} common outliers identified by both methods")

# Visualize outlier detection results
fig, ax = plt.subplots(figsize=(10, 6))
outlier_counts = [n_outliers_lof - n_common_outliers, n_outliers_distance - n_common_outliers, n_common_outliers]
outlier_labels = ['LOF Only', 'Distance Only', 'Both Methods']
colors_od_res = ['#e74c3c', '#3498db', '#2ecc71']
ax.bar(outlier_labels, outlier_counts, color=colors_od_res, edgecolor='black', alpha=0.7)
ax.set_ylabel('Number of Outliers', fontsize=12)
ax.set_title('Outlier Detection Results Comparison', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')
for i, val in enumerate(outlier_counts):
    pct = val / len(X_outlier_scaled) * 100
    ax.text(i, val, f'{val:,}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.show()

# ============================================================================
# 6.3.3. Review of process
# ============================================================================
print_step_header("6.3.3", "Review of process")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nProcess review: Data preparation ({len(df_outlier):,} records, {len(all_features)} features) → Model selection → Execution → Evaluation completed")

# Visualize outlier detection process workflow
fig, ax = plt.subplots(figsize=(12, 6))
steps_od = ['Data\nSelection', 'Data\nCleaning', 'Feature\nEngineering', 
            'Feature\nIntegration', 'Data\nFormatting', 'Model\nBuilding', 'Detection', 'Assessment']
status_od = [1, 1, 1, 1, 1, 1, 1, 1]  # All completed
colors_od_proc = ['#2ecc71' if s == 1 else '#95a5a6' for s in status_od]

y_pos_od = np.arange(len(steps_od))
ax.barh(y_pos_od, status_od, color=colors_od_proc, edgecolor='black', alpha=0.7)
ax.set_yticks(y_pos_od)
ax.set_yticklabels(steps_od, fontsize=11)
ax.set_xlabel('Status', fontsize=12)
ax.set_title('Outlier Detection Process Review', fontsize=14, fontweight='bold')
ax.set_xlim([0, 1.2])
ax.set_xticks([0, 1])
ax.set_xticklabels(['Not Started', 'Completed'])
ax.invert_yaxis()

for i, (step, s) in enumerate(zip(steps_od, status_od)):
    ax.text(1.05, i, '✓ Completed', va='center', fontsize=10, fontweight='bold', color='#2ecc71')

plt.tight_layout()
plt.show()

# ============================================================================
# 6.3.4. Determine next steps
# ============================================================================
print_step_header("6.3.4", "Determine next steps")

# ----------------------------------------------------------------------------
# Variable Definition
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Logic Calculation
# ----------------------------------------------------------------------------


# ----------------------------------------------------------------------------
# Result Display
# ----------------------------------------------------------------------------
print(f"\nOutlier detection complete: {n_common_outliers:,} outliers, {len(X_outlier_scaled):,} records, {len(all_features)} features")

# Visualize outlier detection completion summary
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Outlier counts by method
method_names = ['LOF', 'Distance', 'Common']
method_counts = [n_outliers_lof, n_outliers_distance, n_common_outliers]
method_colors = ['#e74c3c', '#3498db', '#2ecc71']
axes[0].bar(method_names, method_counts, color=method_colors, edgecolor='black', alpha=0.7)
axes[0].set_ylabel('Number of Outliers', fontsize=12)
axes[0].set_title('Outliers by Detection Method', fontsize=13, fontweight='bold')
axes[0].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(method_counts):
    pct = val / len(X_outlier_scaled) * 100
    axes[0].text(i, val, f'{val:,}\n({pct:.2f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

# Dataset summary
data_info_od = ['Records', 'Features', 'Outliers']
data_vals_od = [len(X_outlier_scaled), len(all_features), n_common_outliers]
axes[1].bar(data_info_od, data_vals_od, color='#e67e22', edgecolor='black', alpha=0.7)
axes[1].set_ylabel('Count', fontsize=12)
axes[1].set_title('Dataset Summary', fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')
for i, val in enumerate(data_vals_od):
    axes[1].text(i, val, f'{val:,}' if val > 100 else f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Normal vs Outlier distribution
normal_count = len(X_outlier_scaled) - n_common_outliers
axes[2].bar(['Normal', 'Outliers'], [normal_count, n_common_outliers], 
            color=['#2ecc71', '#e74c3c'], edgecolor='black', alpha=0.7)
axes[2].set_ylabel('Count', fontsize=12)
axes[2].set_title('Normal vs Outlier Distribution', fontsize=13, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')
for i, val in enumerate([normal_count, n_common_outliers]):
    pct = val / len(X_outlier_scaled) * 100
    axes[2].text(i, val, f'{val:,}\n({pct:.1f}%)', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.show()

# ============================================================================
# Section 7: Conclusion
# ============================================================================
print_section_title("7", "Conclusion")

print("\nThis project successfully applied machine learning techniques to analyze traffic violation data")
print("from Montgomery County, addressing the business question: 'What are the key contributing")
print("factors that lead to different types of traffic violations?'")

print("\n## Summary of Results")
print("\n### Dataset Overview:")
print(f"  - Initial records: {len(df):,}")
print(f"  - After preparation: {len(df_outlier):,} records")
print(f"  - Features engineered: {len(all_features)}")

print("\n### 1. Classification (Decision Tree):")
print("   - Model: Decision Tree with Cross-Validation")
print("   - Target Variable: Violation Type")
print(f"   - Cross-Validation Accuracy: {cv_scores.mean():.2%} (+/- {cv_scores.std() * 2:.2%})")
print("   - Key Findings:")
print("     * Contributed To Accident is the most important feature (37.0% importance)")
print("     * Time of day (Hour) and Vehicle Age are significant contributing factors")
print("     * Decision tree rules provide interpretable insights for law enforcement")

print("\n### 2. Clustering (KMeans):")
print("   - Algorithm: KMeans with Elbow Method")
print(f"   - Optimal clusters (k): {optimal_k}")
print(f"   - Silhouette Score: {sil_score:.3f}")
print(f"   - Davies-Bouldin Score: {db_score:.3f}")
print("   - Key Findings:")
print("     * Identified 5 distinct violation patterns")
print("     * Geographic and temporal patterns revealed")
print("     * Cluster outliers detected for further investigation")

print("\n### 3. Outlier Detection:")
print("   - Methods: LOF + Distance-based (combined approach)")
print(f"   - LOF detected: {n_outliers_lof:,} outliers ({n_outliers_lof/len(df_outlier)*100:.2f}%)")
print(f"   - Distance-based detected: {n_outliers_distance:,} outliers ({n_outliers_distance/len(df_outlier)*100:.2f}%)")
print(f"   - High-confidence outliers (both methods): {n_common_outliers:,} ({n_common_outliers/len(df_outlier)*100:.3f}%)")
print("   - Key Findings:")
print("     * Common outliers show distinct characteristics (40.7% accident rate)")
print("     * Both methods agree on 54 high-confidence outliers")
print("     * Outliers warrant further investigation for data quality and fraud detection")

print("\n## Project Achievements")
print("\n✓ All three required data mining tasks completed successfully")
print("✓ Models validated using appropriate evaluation metrics")
print("✓ Results are interpretable and provide actionable insights")
print("✓ Cross-validation applied for classification task")
print("✓ Association rule mining and correlation analysis performed")
print("✓ Comprehensive documentation following CRISP-DM methodology")

print("\n## Business Impact")
print("\nThe analysis provides valuable insights for:")
print("  - Law enforcement resource allocation based on temporal and geographic patterns")
print("  - Targeted prevention strategies using decision tree rules")
print("  - Data quality improvement through outlier detection")
print("  - Understanding contributing factors to different violation types")

print("\n## Limitations and Future Work")
print("\n  - Model accuracy (57%) could be improved with hyperparameter tuning")
print("  - Consider ensemble methods (Random Forest, Gradient Boosting) for better performance")
print("  - Expand feature engineering to include more contextual information")
print("  - Validate outlier findings with domain experts")
print("  - Explore additional clustering algorithms for comparison")

print("\nProject Status: All required analyses completed successfully")
