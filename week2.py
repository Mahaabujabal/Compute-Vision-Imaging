# Step 1: Data Loading & Initial Exploration
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.datasets import load_wine

# تحميل مجموعة بيانات حقيقية مباشرة (Wine Dataset)
wine_data = load_wine()
df = pd.DataFrame(data=wine_data.data, columns=wine_data.feature_names)
df["target"] = wine_data.target

print(df.info())  # فحص أنواع البيانات والشواغر
print(df.describe())  # الملخص الإحصائي


# Step 2: Data Preprocessing & Cleaning
# 1. التعامل مع القيم المفقودة (إن وجدت)
df.dropna(inplace=True)

# 2. فصل الميزات عن الهدف
X = df.drop("target", axis=1)
y = df["target"]


# Step 3: Train-Test Split & Feature Scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# تقسيم البيانات إلى 80% تدريب و 20% اختبار
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# تطبيق StandardScaler لمنع تحيز الأرقام الكبيرة
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 4: Model Training & Evaluation
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss

# تدريب النموذج
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# التنبؤ والتقييم
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_proba))  # تقييم دقة الاحتمالات
