# 📊 Student Grading Dataset Analysis & Preprocessing

This project explores and preprocesses the **[Students Grading Dataset](http://kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset)** from Kaggle using a variety of data preparation techniques. The goal was to clean, transform, and analyze the dataset to prepare it for further machine learning or data analysis tasks.

---

## 📁 Dataset Overview

The dataset contains academic grading records of students, including variables such as:

- Student ID
- Gender
- Grades in different subjects
- Final grade
- Study time
- Failures
- Absences
- ...and more.

📌 [Link to Dataset on Kaggle](http://kaggle.com/datasets/mahmoudelhemaly/students-grading-dataset)

---

## ⚙️ Project Workflow

### 1. 🔍 Data Exploration

- Loaded the dataset using pandas
- Displayed summary statistics and visualized missing values
- Checked for outliers and data distribution

### 2. 🧹 Data Cleaning

- Handled missing values using imputation techniques
- Removed duplicate records
- Standardized column names and categorical entries
- Addressed inconsistent formatting in text fields

### 3. 🔗 Data Integration (Optional)

- *(If any additional data was added, mention the dataset and integration steps here)*

### 4. 📉 Data Reduction Techniques

Implemented multiple reduction techniques to reduce dimensionality and improve model performance:

- **PCA (Principal Component Analysis)**: Reduced the feature space while retaining variance
- **Attribute Subset Selection**: Selected relevant attributes using correlation and feature importance
- **Sampling**: Applied stratified sampling for balanced representation

### 5. 📊 Data Transformation & Feature Engineering

- **Normalization**: Applied Min-Max scaling and Z-score normalization
- **Data Discretization**: Converted numerical scores to grade categories (A, B, C…)
- **Log Transformation**: Applied where skewness was detected
- **Histogram Analysis**: Used to visually assess feature distributions

### 6. 📈 Regression & Modeling (Exploratory)

- **Linear Regression**: To predict final grades based on key features
- **Multiple Regression**: To capture interactions between study time, past failures, and academic outcomes
- **Log-Linear Model**: Modeled categorical and numerical relationships

### 7. 🧠 Clustering (Optional Step)

- Applied K-Means clustering to group students based on academic performance patterns

---

## 📌 Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebook

---

## 📂 Folder Structure

