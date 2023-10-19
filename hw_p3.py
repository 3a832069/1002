import pandas as pd
import rpy2.robjects as robjects
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import StrVector
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix


C50 = importr('C50')

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


# 將目標變量轉換因子
train_y = StrVector([str(label) for label in train_y])
test_y = StrVector([str(label) for label in test_y])

from rpy2.robjects.packages import importr
base = importr('base')


train_y = base.factor(train_y)
test_y = base.factor(test_y)

# 將 NumPy 數組轉換為 Pandas DataFrame

train_x = pd.DataFrame(train_x, columns=combined_data.columns.difference(['income']))
test_x = pd.DataFrame(test_x, columns=combined_data.columns.difference(['income']))

# 創建 C4.5 決策樹模型
with localconverter(robjects.default_converter + pandas2ri.converter):
    train_x_r = robjects.conversion.py2rpy(train_x)
    train_y_r = robjects.conversion.py2rpy(train_y)

c45_model = C50.C5_0(train_x_r, train_y_r)

# 預測
with localconverter(robjects.default_converter + pandas2ri.converter):
    test_x_r = robjects.conversion.py2rpy(test_x)

test_predictions_r = C50.predict_C5_0(c45_model, test_x_r)


#轉換預測為Pyhon數據結構
test_predictions = robjects.conversion.rpy2py(test_predictions_r)
train_predictions_r = C50.predict_C5_0(c45_model, train_x_r)
train_predictions = robjects.conversion.rpy2py(train_predictions_r)

# 將預測結果和真實標籤轉換為 NumPy 數組
test_predictions = np.array(test_predictions)
test_y = np.array(test_y)
train_predictions = np.array(train_predictions)
train_y = np.array(train_y)

# 計算並印出正確率
test_accuracy = accuracy_score(test_y, test_predictions)
print("測試集分類正確率:", test_accuracy)

train_accuracy = accuracy_score(train_y, train_predictions)
print("訓練集分類正確率:", train_accuracy)


#混淆矩阵
test_confusion = confusion_matrix(test_y, test_predictions)
print("測試集混淆矩阵:")
print(test_confusion)

# 計算精確度
precision = precision_score(test_y, test_predictions, pos_label=2)
print("精確度:", precision)

# 計算召回率
recall = recall_score(test_y, test_predictions, pos_label=2)
print("召回率:", recall)


# 計算F1分數
f1 = f1_score(test_y, test_predictions, pos_label=2)
print("F1分數:", f1)

# 創建包含預測結果的DataFrame
test_results = test_data[['income']].copy()
test_results['Predicted'] = test_predictions

# 將結果寫入Excel檔案
test_results.to_excel('test_results.xlsx', index=False)
