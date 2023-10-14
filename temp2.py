# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 18:42:58 2023

@author: linda
"""

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# 讀取資料集
data = pd.read_csv('output.csv')

# 將特徵和目標變數分開
X = data.drop('education', axis=1)  # 使用 'education' 以預測
y = data['education']

# 使用One-Hot編碼處理類別變數
X_encoded = pd.get_dummies(X)

# 切分訓練和測試資料集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

# 定義多個決策樹模型
tree_algorithms = ['id3', 'c4.5', 'c5.0', 'cart']
results = []
"""
best：選擇最佳的分割策略，決策樹會選擇最能區分不同類別的特徵來分割。
random：使用隨機選擇的分割特徵，適用於資料特徵較多時，加速模型訓練。
"""

for algorithm in tree_algorithms:
    # 選擇模型
    if algorithm == 'id3':
        model = DecisionTreeClassifier(criterion='entropy', splitter='best')
    elif algorithm == 'c4.5':
        model = DecisionTreeClassifier(criterion='entropy', splitter='random')
    elif algorithm == 'c5.0':
        model = DecisionTreeClassifier(criterion='gini', splitter='best')
    elif algorithm == 'cart':
        model = DecisionTreeClassifier(criterion='gini', splitter='best')
    else:
        raise ValueError("Unsupported algorithm")

    # 訓練模型
    model.fit(X_train, y_train)

    # 預測
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # 計算訓練資料集和測試資料集的正確率
    train_accuracy = accuracy_score(y_train, y_train_pred)
    test_accuracy = accuracy_score(y_test, y_test_pred)

    results.append({
        'Algorithm': algorithm,
        'Train Accuracy': train_accuracy,
        'Test Accuracy': test_accuracy
    })
    """
    append為對result表新增資料
    """

    # 將分類預測結果輸出到Excel檔案
    output_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_test_pred})
    """
    DataFrame建立表格資料
    '欄位名稱':資料
    """
    output_df.to_excel(f'{algorithm}_predictions.xlsx', index=False)
    
    """
    輸出excel(檔名格式，不要索引)
    
    """

# 輸出結果
result_df = pd.DataFrame(results)
result_df.to_excel('tree_algorithms_performance.xlsx', index=False)

