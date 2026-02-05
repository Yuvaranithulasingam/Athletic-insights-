import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix, classification_report

// LOAD DATA
data = pd.read_csv("athlete_dataset.csv")

// FILL MISSING VALUES (pandas 3.x safe)
data['Age'] = data['Age'].fillna(data['Age'].median())
data['Height'] = data['Height'].fillna(data['Height'].median())
data['Weight'] = data['Weight'].fillna(data['Weight'].median())
data['Sex'] = data['Sex'].fillna('M')

// CREATE MEDAL LABEL
def medal_label(row):
    if row.get('Medal') == 'Gold':
        return 3
    elif row.get('Medal') == 'Silver':
        return 2
    elif row.get('Medal') == 'Bronze':
        return 1
    else:
        return 0
data['Medal_Label'] = data.apply(medal_label, axis=1)

// ENCODE CATEGORICAL VARIABLES
le_sex = LabelEncoder()
le_team = LabelEncoder()
le_noc = LabelEncoder()
le_city = LabelEncoder()

data['Sex'] = le_sex.fit_transform(data['Sex'])
data['Team'] = le_team.fit_transform(data['Team'])
data['NOC'] = le_noc.fit_transform(data['NOC'])
data['City'] = le_city.fit_transform(data['City'])

// SELECT FEATURES
features = ['Sex', 'Age', 'Height', 'Weight', 'Team', 'NOC', 'Year', 'City']
X = data[features]
y_class = data['Medal_Label']
scaler = StandardScaler()
X[['Age','Height','Weight','Year']] = scaler.fit_transform(X[['Age','Height','Weight','Year']])

// COUNTRY-LEVEL TOTAL MEDALS FOR REGRESSION
country_medals = data.groupby(['NOC', 'Year'])['Medal_Label'].sum().reset_index()
y_reg_dict = {(row['NOC'], row['Year']): row['Medal_Label'] for idx, row in country_medals.iterrows()}
average_medals = country_medals['Medal_Label'].mean()
y_reg = X.apply(lambda row: y_reg_dict.get((row['NOC'], row['Year']), average_medals), axis=1)

// SPLIT DATA
X_train, X_test, y_train_cls, y_test_cls = train_test_split(X, y_class, test_size=0.2, random_state=42)
_, _, y_train_reg, y_test_reg = train_test_split(X, y_reg, test_size=0.2, random_state=42)


// RANDOM FOREST CLASSIFIER
rfc = RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42)
rfc.fit(X_train, y_train_cls)
y_pred_cls = rfc.predict(X_test)
print("Random Forest Classifier Accuracy:", accuracy_score(y_test_cls, y_pred_cls))
print("\nConfusion Matrix:\n", confusion_matrix(y_test_cls, y_pred_cls))
print("\nClassification Report:\n", classification_report(y_test_cls, y_pred_cls))

// LINEAR REGRESSION
lr = LinearRegression()
lr.fit(X_train, y_train_reg)
y_pred_reg = lr.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred_reg))
print(f"\n Linear Regression RMSE: {rmse:.2f}")

// SAVE MODELS SAFELY
if not os.path.exists("model"):
    os.makedirs("model")
joblib.dump(rfc, "model/rfc_medal_model.pkl")
joblib.dump(lr, "model/lr_medal_count.pkl")
joblib.dump(scaler, "model/scaler.pkl")  # Save scaler for consistent scaling
print("Models saved successfully ")

// PREDICTION FUNCTION
def predict_athlete(athlete_input):
    """
    athlete_input: [Sex, Age, Height, Weight, Team, NOC, Year, City]
    """
    # Convert to DataFrame with column names
    input_df = pd.DataFrame([athlete_input], columns=features)
    # Scale numerical columns using trained scaler
    input_df[['Age','Height','Weight','Year']] = scaler.transform(input_df[['Age','Height','Weight','Year']]) 
    # Predict medal
    medal_pred = rfc.predict(input_df)[0]
    # Predict country-level total medals with fallback to average
    medal_count = lr.predict(input_df)[0]
    medal_count = max(0, round(medal_count))
    medal_map = {0: "No Medal", 1: "Bronze", 2: "Silver", 3: "Gold"}
    print(f"\nPredicted Medal: {medal_map[medal_pred]}")
    print(f"Expected Total Medals (Country-level): {medal_count}")

// EXAMPLE PREDICTION
# Example input: Male(1), Age=25, Height=180, Weight=75, Team=10, NOC=5, Year=2016, City=3
predict_athlete([1, 25, 180, 75, 10, 5, 2016, 3])
