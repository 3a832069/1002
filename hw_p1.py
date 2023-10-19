# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 21:18:38 2023

@author: linda
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

# 讀取資料集
train_data = pd.read_csv('train.csv')
test_data = pd.read_csv('test.csv')

# 合併訓練和測試數據，以確保相同的熱編碼處理
combined_data = pd.concat([train_data, test_data])

# 創建熱編碼 (One-Hot Encoding)
combined_data = pd.get_dummies(combined_data, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country'])

# 分開特徵和目標變數
train_x = combined_data[:len(train_data)]
train_y = train_x['income']
train_x = train_x.drop('income', axis=1)

test_x = combined_data[len(train_data):]
test_y = test_x['income']
test_x = test_x.drop('income', axis=1)

# 處理缺失值
imputer = SimpleImputer(strategy='most_frequent')
train_x = imputer.fit_transform(train_x)
test_x = imputer.transform(test_x)

# 標準化特徵值
scaler = StandardScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)

# 創建決策樹模型
model = DecisionTreeClassifier()

# 訓練決策樹模型
model.fit(train_x, train_y)

# 進行預測
train_predictions = model.predict(train_x)
test_predictions = model.predict(test_x)

# 計算並印出訓練集的分類正確率
train_accuracy = accuracy_score(train_y, train_predictions)
print("訓練集分類正確率:", train_accuracy)

# 計算並印出測試集的分類正確率
test_accuracy = accuracy_score(test_y, test_predictions)
print("測試集分類正確率:", test_accuracy)

# 計算精確度
precision = precision_score(test_y, test_predictions, pos_label=' >50K')
print("精確度:", precision)

# 計算召回率
recall = recall_score(test_y, test_predictions, pos_label=' >50K')
print("召回率:", recall)

# 計算F1分數
f1 = f1_score(test_y, test_predictions, pos_label=' >50K')
print("F1分數:", f1)


# 計算混淆矩陣
conf_matrix = confusion_matrix(test_y, test_predictions)
print("混淆矩陣:")
print(conf_matrix)
"""
# 創建包含預測結果的DataFrame
test_results = test_data[['income']].copy()
test_results['Predicted'] = test_predictions

# 將結果寫入Excel檔案
test_results.to_excel('test_results.xlsx', index=False)

# 將預處理後的資料存到Excel檔案
preprocessed_train_data = pd.DataFrame(train_x, columns=combined_data.columns.drop('income'))
preprocessed_test_data = pd.DataFrame(test_x, columns=combined_data.columns.drop('income'))
preprocessed_train_data.to_excel('preprocessed_train_data.xlsx', index=False)
preprocessed_test_data.to_excel('preprocessed_test_data.xlsx', index=False)
"""