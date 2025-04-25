import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from preprocess import preprocessor

# Load data
df = pd.read_csv("C:/Users/adabala puja/OneDrive/Desktop/Titanic_survival/tested.csv")

# Feature Engineering
df['Title'] = df['Name'].str.extract(r',\s*([^\.]*)\s*\.', expand=False)
df['Title'] = df['Title'].apply(lambda x: x if x in ['Mr', 'Mrs', 'Miss', 'Master'] else 'Rare')
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop unneeded columns
df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], axis=1, inplace=True)

# Split features and target
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create full pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# Train
pipeline.fit(X_train, y_train)

# Evaluate
y_pred = pipeline.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(pipeline, "../model/titanic_model.pkl")
