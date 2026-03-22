#  Loan Approval Prediction System with Explainable AI

##  Overview

This project focuses on building a machine learning-based system to predict whether a loan application will be approved or not based on applicant details. The goal is to demonstrate how data-driven decision-making can assist financial institutions in evaluating loan eligibility efficiently.

The application is developed as an end-to-end solution, starting from data preprocessing and model training to deployment using an interactive web interface.


##  Key Features

* Predicts loan approval status based on user inputs
* Provides probability score for prediction confidence
* Built using a Random Forest classifier for improved performance
* Interactive user interface using Streamlit
* Includes basic feature-level explanation for better interpretability



##  Tech Stack

* **Programming Language:** Python
* **Libraries & Tools:**

  * Scikit-learn
  * Pandas
  * NumPy
  * Streamlit
  * Matplotlib



##  How It Works

1. The dataset is preprocessed by handling missing values and encoding categorical features.
2. A Random Forest model is trained on the processed data.
3. The trained model is saved and integrated into a Streamlit application.
4. Users can input their details through the interface.
5. The model predicts whether the loan will be approved along with a confidence score.


##  How to Run the Project

1. Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/loan-approval-ml.git
cd loan-approval-ml
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the application:

```bash
streamlit run app.py
```



##  Input Parameters

The model considers the following inputs:

* Gender
* Marital Status
* Dependents
* Education
* Employment Status
* Applicant Income
* Co-applicant Income
* Loan Amount
* Loan Term
* Credit History
* Property Area



## Output

* Loan Approval Status (Approved / Not Approved)
* Prediction Probability (confidence level)



## Future Improvements

* Integration of advanced explainability techniques (SHAP/LIME)
* Model optimization and hyperparameter tuning
* Deployment on cloud platforms for public access


 Author

Koushik
B.Tech (CSE - AI & ML)
Aspiring AI/ML Engineer
