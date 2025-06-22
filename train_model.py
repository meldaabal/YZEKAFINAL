import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Veri setini oku
df = pd.read_csv("insurance.csv")

# One-hot encoding
df = pd.concat([
    df,
    pd.get_dummies(df["sex"], prefix="sex"),
    pd.get_dummies(df["smoker"], prefix="smoker"),
    pd.get_dummies(df["region"], prefix="region")
], axis=1)
df.drop(["sex_female", "smoker_no", "sex", "smoker", "region"], axis=1, inplace=True)
df = df.astype(int)

# X ve y
X = df.drop("charges", axis=1)
y = df["charges"]

# Train/test böl
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Ölçekleme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Model eğit
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)



# model_output klasörünü oluştur
os.makedirs("model_output", exist_ok=True)

# Model ve scaler'ı kaydet
joblib.dump(model, "model_output/random_forest_model.pkl")
joblib.dump(scaler, "model_output/scaler.pkl")

# Eğitim ve test verilerini kaydet
X_train.to_csv("model_output/X_train.csv", index=False)
X_test.to_csv("model_output/X_test.csv", index=False)
y_train.to_csv("model_output/y_train.csv", index=False)
y_test.to_csv("model_output/y_test.csv", index=False)

print("✅ Model ve veriler başarıyla model_output klasörüne kaydedildi.")

model.feature_names_in_ = X.columns.to_numpy()
print("📌 Modelin gördüğü sütun sırası:")
print(model.feature_names_in_)
