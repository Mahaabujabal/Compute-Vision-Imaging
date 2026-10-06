# Step 1: Data Loading & Initial Exploration
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

df = pd.read_csv("your_downloaded_dataset.csv")

print(df.info())  # فحص أنواع البيانات والشواغر
print(df.describe())  # الملخص الإحصائي



# Step 2: Data Preprocessing & Cleaning
# 1. التعامل مع القيم المفقودة
df.dropna(inplace=True)

# 2. تحويل المتغيرات النصية (Categorical Encoding)
df = pd.get_dummies(df, drop_first=True)

# 3. فصل الميزات عن الهدف
X = df.drop("Target_Column", axis=1)
y = df["Target_Column"]


# Step 3: Train-Test Split & Feature Scaling
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# تطبيق StandardScaler لمنع تحيز الأرقام الكبيرة
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# Step 4: Model Training & Evaluation
from sklearn.metrics import accuracy_score, classification_report, log_loss
from sklearn.linear_model import LogisticRegression

# تدريب النموذج
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# التنبؤ والتقييم
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("Log Loss:", log_loss(y_test, y_proba))  # تقييم دقة الاحتمالات
