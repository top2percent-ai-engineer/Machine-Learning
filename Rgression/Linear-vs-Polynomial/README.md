# Multiple & Polynomial Regression – Position Salaries Project

## 📌 Project Overview
This project demonstrates the implementation and comparison of **Multiple Linear Regression** and **Polynomial Regression** models using the **Position_Salaries dataset**.  
The objective is to analyze how salary varies with job level and determine which regression model best fits the data.

---

## 🎯 Objectives
- Load and explore the Position_Salaries dataset
- Perform basic Exploratory Data Analysis (EDA)
- Implement Multiple Linear Regression
- Implement Polynomial Regression (degree = 2)
- Evaluate models using **R² Score**
- Compare model performance and visualize results

---

## 📂 Dataset
**Position_Salaries.csv**

### Columns:
- `Position` – Job title
- `Level` – Numeric job level
- `Salary` – Corresponding salary

> Dataset provided by instructor

---

## 🛠️ Technologies & Libraries Used
- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn

---

## 🧪 Steps Performed

### 1️⃣ Data Loading
- Dataset loaded using `pandas.read_csv()`

### 2️⃣ Exploratory Data Analysis (EDA)
- `.info()` for data types
- `.describe()` for statistical summary
- Missing value detection using `.isnull()`

### 3️⃣ Multiple Linear Regression
- Feature: `Level`
- Target: `Salary`
- 80/20 train-test split
- Model trained using `LinearRegression`
- Coefficients and intercept extracted

### 4️⃣ Model Evaluation
- Predictions on test data
- Performance evaluated using **R² Score**
- Actual vs Predicted plot

### 5️⃣ Polynomial Regression
- Feature transformation using `PolynomialFeatures(degree=2)`
- Linear model trained on polynomial features
- R² Score calculated
- Regression curve plotted for visualization

---

## 📊 Model Comparison

| Model | R² Score |
|------|---------|
| Multiple Linear Regression | Lower |
| Polynomial Regression | Higher |

### ✔ Conclusion
Polynomial Regression performs better because the relationship between **job level and salary is non-linear**, which cannot be captured effectively by a straight-line model.

---

## ▶ How to Run the Project
1. Clone the repository
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
