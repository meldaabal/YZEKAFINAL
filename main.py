import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Lasso, Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Tüm sütunların çıktıda tam görünmesi için ayar
pd.set_option('display.max_columns', None)

# Veri setini oku
df = pd.read_csv("insurance.csv")

# Veri ön inceleme
print(df.head())
print(df.describe())
print(df.isnull().sum())

# Kategorik verilerin dağılımı
print("Yaş dağılımı:\n", df["age"].value_counts())
print("Cinsiyet dağılımı:\n", df["sex"].value_counts())
print("BMI dağılımı:\n", df["bmi"].value_counts())
print("Çocuk sayısı dağılımı:\n", df["children"].value_counts())
print("Sigara içme durumu:\n", df["smoker"].value_counts())
print("Bölge dağılımı:\n", df["region"].value_counts())

# Kategorik verileri sayısal hale getirme (One-Hot Encoding)
sex_dummies = pd.get_dummies(df["sex"], prefix="sex")
smoker_dummies = pd.get_dummies(df["smoker"], prefix="smoker")
region_dummies = pd.get_dummies(df["region"], prefix="region")

df = pd.concat([df, sex_dummies, smoker_dummies, region_dummies], axis=1)
df.drop(["sex_female", "smoker_no", "sex", "smoker", "region"], axis=1, inplace=True)
df = df.astype(int)

print(df.head())
print(df.dtypes)

# Giriş (X) ve çıkış (y)
X = df.drop("charges", axis=1)
y = df["charges"]

# Eğitim ve test ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
print("\n🎯 Eğitim ve Test Verisi Boyutları:")
print("X_train:", X_train.shape)
print("y_train:", y_train.shape)
print("X_test :", X_test.shape)
print("y_test :", y_test.shape)

# Ölçekleme (standardizasyon)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ✨ İlk 5 gözlemi yorumlu göster
print("\n📌 X_train ilk 5 satır (ölçeklenmiş):")
scaled_df = pd.DataFrame(X_train_scaled[:5], columns=X.columns)
print(scaled_df)

print("\n🧠 Açıklama:")
print("- Z-score kullanılarak standardize edilmiştir.")
print("- 0 = Ortalama, < 0 = Ortalamanın altında, > 0 = Ortalamanın üstü")

# MODELLER
models = {
    "Karar Ağacı": DecisionTreeRegressor(max_depth=2, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Lasso": Lasso(alpha=1.0),
    "Ridge": Ridge(alpha=1.0),
    "ElasticNet": ElasticNet(alpha=1.0)
}

# Eğitim + Tahmin
predictions = {}
for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    predictions[name] = preds

# ✨ İlk 5 kayıt için her modelin tahminleri + açıklamalı
print("\n🧾 MODELLERE GÖRE: Giriş Açıklamaları + Tahminler (İlk 5 Satır):\n")
for i in range(5):
    print(f"📦 Kayıt {i+1}:")
    for j, col in enumerate(X.columns):
        val = X_train_scaled[i][j]
        yorum = ""
        if col == "age":
            yorum = "Genç" if val < 0 else "Yaşlı"
        elif col == "bmi":
            yorum = "Zayıf" if val < 0 else "Kilolu"
        elif col == "children":
            yorum = "Az çocuk" if val < 0 else "Çok çocuk"
        elif col == "sex_male":
            yorum = "Erkek" if val > 0 else "Kadın"
        elif col == "smoker_yes":
            yorum = "Sigara içiyor" if val > 0 else "Sigara içmiyor"
        elif "region" in col:
            yorum = "Bu bölgede yaşıyor" if val > 0 else "Hayır"
        print(f"{col:18}: {val:>6.2f} → {yorum}")
    for name in models:
        tahmin = models[name].predict([X_train_scaled[i]])[0]
        print(f"🔮 {name:14} Tahmini: {int(tahmin)} TL")
    print("-" * 60)

# 📊 HATA METRİKLERİ + YORUMLU SONUÇ TABLOSU
results = pd.DataFrame(columns=["MAE", "MSE", "RMSE", "R2", "Yorum"])
for name in models:
    mae = mean_absolute_error(y_test, predictions[name])
    mse = mean_squared_error(y_test, predictions[name])
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, predictions[name])

    if mae < 3000 and r2 > 0.80:
        yorum = "✅ İyi tahmin gücü"
    elif mae < 4500 and r2 > 0.70:
        yorum = "⚠️ Kabul edilebilir"
    else:
        yorum = "❌ Zayıf performans"

    results.loc[name] = [mae, mse, rmse, r2, yorum]

print("\n📈 TÜM MODELLERİN PERFORMANSI + YORUM:")
print(results.round(2))

