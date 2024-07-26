import os
import pandas as pd
from sklearn.model_selection import train_test_split

from extract_features import extract_features


fall_path = 'raw_dataset/Fall/Train/'
adl_path = 'raw_dataset/ADL/Train/'

fall_files = [os.path.join(fall_path, f) for f in os.listdir(fall_path) if f.endswith('.csv')]
adl_files = [os.path.join(adl_path, f) for f in os.listdir(adl_path) if f.endswith('.csv')]

features = []
labels = []

for file in fall_files:
    feature_vector = extract_features(file)
    features.append(feature_vector)
    labels.append(1)  #fall

for file in adl_files:
    feature_vector = extract_features(file)
    features.append(feature_vector)
    labels.append(0)  #ADL

df = pd.DataFrame(features, columns=[
    'sv_max', 'sv_range', 'num_peaks', 'avg_peak_height', 'avg_distance_between_peaks'
])
df['fall'] = labels

X = df.drop('fall', axis=1)
y = df['fall']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X_train.to_csv('dataset/train_features.csv', index=False)
y_train.to_csv('dataset/train_labels.csv', index=False)
X_test.to_csv('dataset/test_features.csv', index=False)
y_test.to_csv('dataset/test_labels.csv', index=False)

print("Training and test datasets created and saved successfully.")