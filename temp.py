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
from sklearn import tree
import graphviz

# 讀取資料集
data = pd.read_csv('output.csv',)
#print(data.head()) 印出資料

# 將特徵和目標變數分開
X = data.drop('education', axis=1)  # 使用 'education' 以預測
"""
axis=1代表水平方向(列)
axis=0代表垂直方向(行)
"""
y = data['education']

# 使用One-Hot編碼處理類別變數
"""
目前特徵為非數值的資料，因此需要進行編碼才能跑計算
"""

X_encoded = pd.get_dummies(X)

# 切分訓練和測試資料集
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)
"""
test_size=0.2 為將資料分成訓練集和測試集時，測試集的大小為整個資料集的20%
將編碼後的特徵資料 X_encoded 和目標資料 y 分割為
訓練集（X_train 和 y_train）和測試集（X_test 和 y_test）
random_state相同就可獲得相同的隨機結果，可以是任何整數
"""
# 選擇模型（這裡使用決策樹）
model = DecisionTreeClassifier()

# 訓練模型
model.fit(X_train, y_train)

# 預測
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)
"""
用訓練好的特徵x預測目標變數Y
存起來用來比較訓練資料與測試資料
"""

# 計算訓練資料集和測試資料集的正確率
train_accuracy = accuracy_score(y_train, y_train_pred)
test_accuracy = accuracy_score(y_test, y_test_pred)



# 輸出結果
print("訓練資料集分類正確率：", train_accuracy)
print("測試資料集分類正確率：", test_accuracy)


# 輸出圖片
dot_data = tree.export_graphviz(model, out_file=None, 
                           feature_names=X_encoded.columns,
                           class_names=y.unique(),
                           filled=True, rounded=True, special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("decision_tree", format="pdf")  
#Image(graph.render("decision_tree.pdf"))
"""
filled:顏色填充
rounded:圓角
X_encoded.columns:特徵欄位名稱
y.unique()可視化類別名稱??
"""










