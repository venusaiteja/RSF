import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load dataset
df = pd.read_csv("df_sql.csv")

# Convert Date column
df["Date"] = pd.to_datetime(df["Date"], dayfirst=True)
df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month
df["Week"] = df["Date"].dt.isocalendar().week

# Drop unwanted columns
df = df.drop(["Date"], axis=1)

# Convert categorical column
df["Type"] = df["Type"].astype("category").cat.codes

# Fill missing values
df = df.fillna(0)

# Features and target
X = df.drop("Weekly_Sales", axis=1)
y = df["Weekly_Sales"]

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=10)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open("model.pkl", "wb"))

print("Model trained and saved successfully!")