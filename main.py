# import libraries
from pyarrow import csv
import glob
import pandas as pd
import numpy as np
import sys
import sklearn
import io
import random
import matplotlib.pyplot as plt
from keras.layers import Convolution1D, Dense, Dropout, Flatten, MaxPooling1D, GRU
from sklearn.preprocessing import MinMaxScaler, StandardScaler, LabelEncoder, Normalizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, precision_recall_curve, auc, average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Model, load_model

#data = pd.read_csv("/content/drive/MyDrive/MLinCybersecurity/train_data_10.csv")
data = pd.read_csv("/content/drive/MyDrive/MLinCybersecurity/train.csv")
# 1. Loại bỏ các hàng chứa giá trị NaN hoặc điền giá trị trung bình
data_cleaned = data.fillna(0)  # Hoặc .dropna()

# 2. Mã hóa cột 'Label' (target)
label_encoder = LabelEncoder() # Initialize LabelEncoder
data_cleaned['Label'] = label_encoder.fit_transform(data_cleaned['Label'])
num_classes = len(label_encoder.classes_)
target_strings = label_encoder.inverse_transform(np.arange(num_classes))


# 3. Chia thành X (features) và y (label)
X = data_cleaned.drop(columns=['ID', 'Label'])  # Bỏ cột không cần thiết
y_test = data_cleaned['Label']

# 4. Chuẩn hóa dữ liệu (StandardScaler)
scaler = StandardScaler()
X_test = scaler.fit_transform(X)

# 2. Đánh giá trên tập test
loaded_model = load_model('./Model/conv1d_model_v2-100percent.keras')
y_pred_prob = loaded_model.predict(X_test)  # Get predicted probabilities

y_pred = np.argmax(y_pred_prob, axis=1)  # Get predicted labels
# Ánh xạ các chỉ số phân loại thành nhãn gốc
mapped_y_pred = label_encoder.inverse_transform(y_pred)

# Ánh xạ các nhãn thực tế thành nhãn gốc
mapped_y_test = label_encoder.inverse_transform(y_test)

# In báo cáo phân loại (classification report)
report = classification_report(mapped_y_test, mapped_y_pred)
print(report)
