# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
"""print('hello world')
print('hi')
print('hiiiii')"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# 讀取資料集
data = pd.read_csv('output.csv',)
#print(data.head()) 印出資料

# 將特徵和目標變數分開
X = data.drop('education', axis=1)  # 使用 'education' 以預測
y = data['education']

# 使用One-Hot編碼處理類別變數
X_encoded = pd.get_dummies(X)

# 切分訓練和測試資料集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 選擇模型（這裡使用決策樹）
model = DecisionTreeClassifier()

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 計算訓練資料集和測試資料集的正確率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)

# 輸出結果
print("訓練資料集分類正確率：", train_accuracy)
print("測試資料集分類正確率：", test_accuracy)













