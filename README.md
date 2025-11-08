# Medical Insurance Cost Prediction

[![Python](https://img.shields.io/badge/Python-3.7%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Project Overview

This project demonstrates an end-to-end machine learning workflow to predict individual medical insurance costs. By analyzing key factors such as age, BMI, smoking status, and region, the project builds a robust predictive model and deploys it as an interactive web application.

**Domain:** Healthcare and Insurance  
**Skills Demonstrated:** Python, Machine Learning, Exploratory Data Analysis (EDA), Data Preprocessing, Streamlit, MLflow

## Table of Contents

- [Project Overview](#project-overview)
- [Table of Contents](#table-of-contents)
- [Problem Statement](#problem-statement)
- [Methodology](#methodology)
- [Key Findings from EDA](#key-findings-from-eda)
- [Modeling](#modeling)
- [MLflow Integration](#mlflow-integration)
- [Streamlit Web Application](#streamlit-web-application)
- [Project Structure](#project-structure)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Usage](#usage)
- [Results](#results)
- [Challenges & Solutions](#challenges--solutions)
- [Author](#author)

## Problem Statement

The goal is to build a regression model that accurately estimates medical insurance premiums for individuals based on their personal characteristics and health metrics. This helps insurers understand cost drivers and allows individuals to estimate potential expenses.

## Methodology

1.  **Data Loading & Preprocessing:** Loaded the `medical_insurance.csv` dataset. Verified data quality (found to be clean with no missing values or duplicates). Performed feature engineering by creating interaction terms (e.g., `age_smoker`, `bmi_smoker`) based on EDA insights.
2.  **Exploratory Data Analysis (EDA):** Conducted thorough analysis to understand the distribution of features and their relationships with the target variable (`charges`). Identified key cost drivers.
3.  **Modeling:** Trained and evaluated multiple regression models (Linear Regression, Random Forest Regressor) using Scikit-learn.
4.  **Evaluation:** Compared models using metrics like R-squared (R2), Root Mean Squared Error (RMSE), and Mean Absolute Error (MAE).
5.  **Model Tracking & Management:** Integrated MLflow to track experiments, log parameters and metrics, and register the best-performing model.
6.  **Deployment:** Built an interactive Streamlit web application to serve the trained model, allowing users to input their details and receive cost predictions.

## Key Findings from EDA

- **Smoking Status:** The strongest predictor of high insurance costs. Smokers pay significantly more than non-smokers.
- **Age:** Shows a strong positive correlation with charges. Older individuals generally face higher premiums.
- **BMI:** Also positively correlated with charges, especially when combined with smoking status.
- **Gender & Region:** Have relatively minor impacts on costs compared to smoking, age, and BMI.
- **Children:** Number of children has a very weak positive correlation with charges.

## Modeling

- **Algorithms:** Linear Regression, Random Forest Regressor.
- **Performance Comparison:**
  - **Linear Regression:** R2 Score: ~0.83, RMSE: ~$5112
  - **Random Forest Regressor:** R2 Score: ~0.95, RMSE: ~$2761
- **Selected Model:** **Random Forest Regressor** due to its superior performance in capturing non-linear relationships and complex interactions in the data.

## MLflow Integration

- **Purpose:** To manage the ML lifecycle, including experiment tracking and model registry.
- **Actions:**
  - Tracked model parameters (e.g., `n_estimators`, `random_state`), performance metrics (R2, RMSE, MAE), and the model artifact itself for each training run.
  - Registered the best-performing Random Forest model in the MLflow Model Registry under the name `Medical_Insurance_Cost_Predictor`.

## Streamlit Web Application

- **Functionality:** Provides a user-friendly interface for cost prediction.
- **Features:**
  - Interactive sliders and dropdowns for user input (age, sex, BMI, children, smoker, region).
  - Displays the user's input parameters.
  - Loads the registered Random Forest model from MLflow and displays the predicted insurance cost.
  - Shows static EDA visualizations (e.g., distribution of charges, impact of smoking) for context.

## Technologies Used

- **Programming Language:** Python
- **Data Analysis & Manipulation:** Pandas, NumPy
- **Machine Learning:** Scikit-learn
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Model Management & Tracking:** MLflow
- **Web Framework:** Streamlit

## Getting Started

### Prerequisites

- Python 3.7 or higher installed.
- `pip` package installer for Python.

### Installation

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/B4Mprofessor/medical-insurance-cost-prediction.git
    cd medical-insurance-cost-prediction
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    # Using venv (Python standard library)
    python -m venv venv

    # Activate the virtual environment
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Exploratory Data Analysis (EDA):**

    - Open `notebooks/01_eda.ipynb` in Jupyter and run the cells.

2.  **Model Training & MLflow Integration:**

    - Open `notebooks/02_model_training.ipynb`.
    - Run cells to load data, preprocess, train models, evaluate, and log/register the best model to MLflow.

3.  **Run the Streamlit Web Application:**
    - Ensure you are in the project root directory and your virtual environment is activated.
    - Run the following command:
      ```bash
      streamlit run app.py
      ```
    - The application should open in your default web browser (usually `http://localhost:8501`).

## Results

- The Random Forest model achieved a high R-squared score (~0.95), indicating it explains a large portion of the variance in insurance costs.
- The model demonstrates strong predictive power with a relatively low RMSE (~$2761).
- The Streamlit app provides a functional and interactive interface for users to estimate insurance costs based on their inputs.

## Challenges & Solutions

- **Challenge:** Difficulty loading the MLflow model in the Streamlit app due to path inconsistencies.
  - **Solution:** Explicitly set the MLflow tracking URI in `app.py` to point to the local `mlruns` directory relative to the script's location.
- **Challenge:** Schema enforcement error in MLflow when making predictions due to data type mismatches (e.g., integer vs. boolean for encoded features).
  - **Solution:** Ensured the preprocessing pipeline in `app.py` explicitly converted relevant encoded columns to the `bool` data type before passing them to the model.

---

_This project was created as part of a machine learning capstone project to demonstrate end-to-end model development and deployment skills._
