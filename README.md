# Titanic Survival Prediction

📋 Task Objectives

  1.Build a Machine Learning model to predict whether a passenger survived the Titanic disaster.
  
  2.Handle missing values, encode categorical variables, and normalize numerical features.
  
  3.Train, validate, and test multiple classification models.
  
  4.Evaluate performance using metrics like Accuracy, Precision, Recall, and F1 Score.
  
  5.Deliver clean, modular code with proper preprocessing and pipeline integration.

🛠️ Project Structure:
      Titanic_survival/
          ├── data/
          │   ├── train.csv         # Training dataset
          │   ├── test.csv          # Test dataset
          │
          ├── model/
          │   ├── titanic_model.pkl # Saved trained model
          │
          ├── src/
          │   ├── preprocess.py     # Preprocessing functions (handling missing values, encoding, scaling)
          │   ├── train_model.py    # Model training, evaluation, and saving
          │
          ├── README.md             # Project overview and instructions
          ├── requirements.txt      # Required Python libraries

⚙️ Steps to Run the Project:

    1.Clone the Repository:
    
          git clone https://github.com/Pujithasri3112/Titanic_survival.git
          cd Titanic_survival
    
    2.Create a Virtual Environment:
    
          python -m venv venv
    
    3.Activate the Virtual Environment:

          venv\Scripts\activate
    
    4.Install the Dependencies:
    
          pip install -r requirements.txt
    
    5.Train the Model:
    
          cd src
          python train_model.py
    
    .The trained model will be saved as model/titanic_model.pkl.

🧹 Preprocessing Steps

>Missing Values

Age, Fare: Imputed with the median.

Embarked: Imputed with the most frequent value.

Cabin: Missing cabins marked as "Unknown".

Feature Encoding

Categorical variables like Sex, Embarked are encoded using One-Hot Encoding.

Feature Scaling

Numerical features are scaled using StandardScaler.

Pipeline

A complete preprocessing and modeling pipeline is built using scikit-learn Pipelines for clean execution.


