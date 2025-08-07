# Medical Insurance Cost Prediction

This project predicts medical insurance costs using machine learning. It includes data exploration, model training with MLflow tracking, and a Streamlit web application for cost estimation based on user input.

## Project Overview

This project aims to build an end-to-end Machine Learning regression model to predict individual medical insurance costs. Using Python, we analyze key factors influencing insurance premiums, such as age, gender, Body Mass Index (BMI), number of dependents (children), smoking status, and geographical region.

The goal is to develop an accurate and reliable model that can estimate insurance charges, providing valuable insights for both insurance companies and individuals.

## Features

- **Data Analysis:** Exploratory Data Analysis (EDA) to understand relationships between features and insurance costs.
- **Machine Learning Modeling:** Training and evaluation of regression models (Linear Regression, Random Forest) using Scikit-learn.
- **Model Experiment Tracking:** Integration with MLflow to track experiments, log parameters, metrics, and manage different versions of the model.
- **Interactive Web App:** A user-friendly Streamlit application where users can input their details and get a prediction of their insurance cost.
- **Visual Insights:** Display of key EDA findings within the Streamlit app.

## Technologies Used

- **Programming Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn
- **Model Management:** MLflow
- **Web Framework:** Streamlit
- **Visualization:** Plotly

## Project Structure

medical-insurance-cost-prediction/
│
├── data/
│ ├── raw/
│ │ └── medical_insurance.csv # Original dataset
│ └── processed/
│ └── medical_insurance_processed.csv # Processed dataset (created by notebook)
│
├── notebooks/
│ ├── 01_eda.ipynb # Exploratory Data Analysis
│ └── 02_model_training.ipynb # Model Training, Evaluation, and MLflow Integration
│
├── mlruns/ # MLflow local tracking directory (created after running notebooks)
│
├── app.py # Main Streamlit web application file
│
├── requirements.txt # List of required Python packages (create this, see below)
│
└── README.md # This file

## Getting Started

These instructions will get you a copy of the project up and running on your local machine.

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
    Creating a virtual environment helps isolate project dependencies.

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
    Install the necessary Python libraries using `pip`.
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn mlflow streamlit plotly
    ```
    _(Alternatively, if you create a `requirements.txt` file listing these packages, you can use `pip install -r requirements.txt`)_

### Running the Project

1.  **Exploratory Data Analysis (EDA):**

    - Open the Jupyter Notebook: `notebooks/01_eda.ipynb`.
    - Run the cells to perform data exploration and understand the dataset.

2.  **Model Training and MLflow Integration:**

    - Open the Jupyter Notebook: `notebooks/02_model_training.ipynb`.
    - Run the cells to preprocess data, train models (Linear Regression, Random Forest), evaluate them, log experiments to MLflow, and register the best model (`Random Forest`) in the MLflow Model Registry.
    - This step will create the `mlruns` directory and populate it with experiment data.

3.  **Run the Streamlit Web Application:**
    - Ensure you are in the main project directory (`medical-insurance-cost-prediction`).
    - Run the following command in your terminal:
      ```bash
      streamlit run app.py
      ```
    - Your default web browser should open automatically, displaying the application interface (usually at `http://localhost:8501`).

## Usage

1.  Open the Streamlit app in your browser.
2.  Use the sliders and dropdown menus in the sidebar to input your details (Age, Sex, BMI, Number of Children, Smoker Status, Region).
3.  The application will display:
    - Your selected input parameters.
    - The predicted insurance cost based on the trained Random Forest model.
    - Static visualizations showing key insights from the Exploratory Data Analysis (e.g., impact of smoking, age, BMI on costs).

## Model Performance

The Random Forest Regressor was selected as the best performing model based on evaluation metrics:

- **R2 Score:** ~0.95
- **RMSE:** ~$2761
- **MAE:** ~$1325

The model is tracked and managed using MLflow.

## Author

Abhik Sahana

## Domain

Healthcare and Insurance
