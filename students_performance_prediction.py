# -*- coding: utf-8 -*-
"""Students_Performance_Prediction.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/11gHScuQuUDLv8wNO753ygdvP6WHZ3ylQ
"""

!pip install ucimlrepo
!pip install optuna

"""# Student Performance Prediction

- Author  : Muhammad Aditya Bayhaqie
- Assignment : Machine Learning Terapan (Dicoding)

### Data and Library Loading

Import Library
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# %matplotlib inline
import seaborn as sns
from sklearn.preprocessing import OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

"""Import Dataset"""

from ucimlrepo import fetch_ucirepo

# fetch dataset
predict_students_dropout_and_academic_success = fetch_ucirepo(id=697)

# data (as pandas dataframes)
X = predict_students_dropout_and_academic_success.data.features
y = predict_students_dropout_and_academic_success.data.targets

# metadata
print(predict_students_dropout_and_academic_success.metadata)

# variable information
display(predict_students_dropout_and_academic_success.variables)
# After the information displayed, it is advised if you see it on table view if you use google collab

"""Dari deskripsi data ini, kita bisa simpulkan bahwa hanya ada 2 data Kontinu yang bisa kita analisa apakah ada data outlier nantinya"""

# load the dataset
url = 'https://archive.ics.uci.edu/static/public/697/data.csv'
students = pd.read_csv(url)
students

null_counts = students.isnull().sum()

# Print the null counts
print(null_counts)

# Total number of null values in the DataFrame
total_nulls = null_counts.sum()
print(f"\nTotal number of null values: {total_nulls}")

"""Data sudah clean, which means we can go directly to the coding process

### Exploratory Data Analysis
"""

students.info()

students.describe()

"""3 Nilai x,y,z dominan kosong, sehingga kemungkinan data ini bakalan kita drop

#### Data Outlier
Mari kita liat Data Outlier pada data kontinu yang kita temukan pada deskripsi data terkait
"""

specific_cols = ['Previous qualification (grade)', 'Admission grade']
for col in specific_cols:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=students[col])  # Using students[col] to access data
    plt.title(f'Boxplot of {col}')
    plt.show()

"""Cukup banyak Outlier pada data ini, dalam hal ini kita bakalan drop data Outlier ini dengan memberikan parameter data yang akan kita gunakan"""

# Specify the columns to check for outliers
cols_to_check = ['Previous qualification (grade)', 'Admission grade']

# Calculate quantiles and IQR for the specified columns only
Q1 = students[cols_to_check].quantile(0.25)
Q3 = students[cols_to_check].quantile(0.75)
IQR = Q3 - Q1

# Filter the dataset to remove outliers in the specified columns
filtered_students = students[~((students[cols_to_check] < (Q1 - 1.5 * IQR)) | (students[cols_to_check] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Print the shape of the filtered dataset
print(filtered_students.shape)

"""#### Univariate Analysis
Mari kita cek data ini secara terpisah berdasarkan jenisnya, yaitu kategorikal dan numerikal
"""

categorical_features = students.select_dtypes(include='object').columns.tolist()

numerical_features   = students.select_dtypes(include=['int64', 'float64']).columns.tolist()
numerical_features   = [col for col in numerical_features if col != 'Id' and col != 'SalePrice']

discrete_features    = [col for col in numerical_features if len(students[col].unique()) < 25]
continuous_feature  = [col for col in numerical_features if col not in discrete_features]

print(f'Number of Categorical Feature : {len(categorical_features)}')
print(f'Number of Numerical Feature   : {len(numerical_features)}')
print(f'Number of Discrete Feature    : {len(discrete_features)}')
print(f'Number of Continous Feature   : {len(continuous_feature)}')

display(continuous_feature)

"""Data data yang ada pada data kontinu ini merupakan data yang sebelumnya telah di Encode, data kontinu yang sedari awal digunakan ialah `Previous qualification (grade)` dan `Admission grade` yang sudah kita drop tadi data outliernya.

---

Mari kita analisa terlebih dahulu categorical features

##### Categorical Features
"""

feature = categorical_features[0]
count = students[feature].value_counts()
percent = 100*students[feature].value_counts(normalize=True)
df = pd.DataFrame({'Jumlah Sampel':count, 'Persentase':percent.round(1)})
print(df)
count.plot(kind='bar', title=feature);

"""Data didominasi oleh "Graduate", dimana dari data kita bisa simpulkan bahwa Siswa memiliki tingkat kelulusan yang relatif tinggi

##### Numerical Features
Next, lets analyze the Numerical
"""

students.hist(bins=50, figsize=(20,15))
plt.show()

skewness = students[numerical_features].skew().sort_values(ascending=False)

avg_skewness = skewness
avg_skewness = avg_skewness.sort_values(ascending=False)

print(avg_skewness)

"""Informasi yang mampu kita dapatkan adalah
- Fitur Kontinu relatif Middle Skewed
- Data data lain persebarannya cukup random, karena secara besar data number yang kita dapatkan adalah data Encoded

#### Multivariate Analysis
Mari kita cek keterkaitan antar data ini satu sama lain!

##### Categorical Features
Since we only have 1 Categorical Features, lets proceed into the Numerical Features!

##### Numerical Features
Next up, mari kita cek keterkaitan antar Fitur Numerikal pada data
"""

# Mengamati hubungan antar fitur numerik dengan fungsi pairplot()
sns.pairplot(students, diag_kind = 'kde')

"""Mari kita perdalam analisa kita dengan menggunakan visualisasi _HeatMap_"""

plt.figure(figsize=(20, 16))
correlation_matrix = students[numerical_features].corr().round(2)

# Untuk menge-print nilai di dalam kotak, gunakan parameter anot=True
sns.heatmap(data=correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5, )
plt.title("Correlation Matrix untuk Fitur Numerik ", size=20)

"""Dari hasil analisa dari 2 Visualisasi, kita bisa simpulkan bahwa

- Fitur dengan awalan `Curricular units 1st` dan `Curricular units 2nd` memiliki korelasi yang tinggi
- Fitur dengan Korelasi yang rendah adalah :
  - Marital Status
  - Nationality
  - Displaced
  - Father's occupation
  - Mother's occupation
  - International
  - Unemployment rate
  - Inflation rate
  - GDP
  
  Untuk Fitur ini, akan kita drop

-  Beberapa fitur memiliki korelasi yang tinggi satu sama lain, yaitu
  - Previous Qualification & Admission Grade
  - Age at Enrollment & Scholarship Holder

  Untuk fitur ini, neither akan kita drop salah satu nya, atau akan kita reduce dimensionalitynya
"""

# Drop specified features
features_to_drop = ['Marital status', 'Nationality', 'Displaced', "Father's occupation", "Mother's occupation", 'International', 'Unemployment rate', 'Inflation rate', 'GDP']
students = students.drop(columns=features_to_drop, errors='ignore')

# Show the first few rows of the modified DataFrame
display(students)

"""Buat fitur `Previous Qualification` & `Admission Grade`, bakalan kita drop salah satunya, in this case, we will drop `Previous Qualification`"""

# Drop specified features
features_to_drop = ['Previous qualification (grade)']
students = students.drop(columns=features_to_drop, errors='ignore')

# Show the first few rows of the modified DataFrame
display(students)

"""### Data Preparation

#### Dimension Reduction with Principal Component Analysis (PCA)
Seperti yang kita analisa sebelumnya, fitur dengan nama`Curricular units 1st` dan `Curricular units 2nd` akan kita gabungkan menggunakan **PCA** secara terpisah

Curricular units 1st
"""

# Select columns for PCA
curricular_units_1st_cols = [col for col in students.columns if 'Curricular units 1st' in col]
curricular_units_2nd_cols = [col for col in students.columns if 'Curricular units 2nd' in col]

# Create a DataFrame with selected columns
curricular_units_data = students[curricular_units_1st_cols + curricular_units_2nd_cols]

# Visualize relationships using pairplot
sns.pairplot(curricular_units_data, plot_kws={"s": 3});

# Scale the data before applying PCA
scaler = StandardScaler()
curricular_units_scaled = scaler.fit_transform(curricular_units_data)

# Apply PCA with 3 components (adjust as needed)
pca = PCA(n_components=7, random_state=123)
pca.fit(curricular_units_scaled)
principal_components = pca.transform(curricular_units_scaled)

# Explained variance ratio
explained_variance = pca.explained_variance_ratio_.round(2)
print("Explained Variance Ratio:", explained_variance)

"""Dari hasil PCA ini, kita bisa nyimpulin bahwa 3 Component pertama memiliki Ratio mencapai 90%

Maka step yang akan kita lakukan ialah :

- Gunakan n_component = 3, karena kita bakalan ekstrak 3 Komponen untuk reduksi data.
- Fit model dengan data masukan.
- Tambahkan fitur baru ke dataset dengan nama 'dimension' dan lakukan proses transformasi.
- Drop kolom dengan nama `Curricular units 1st` dan `Curricular units 2nd`
"""

# Create new columns in the 'students' DataFrame for principal components
for i in range(3):
    students[f'Curricular_Units_PCA_{i+1}'] = principal_components[:, i]

# Drop the original Curricular units columns
students = students.drop(columns=curricular_units_1st_cols + curricular_units_2nd_cols)

display(students)

"""Next, untuk `Age at Enrollment` & `Scholarship Holder`

Fitur ini punya makna yang berbeda, so we let it be

#### Train-Test-Split
Disini kita bakal ngambil Ratio 8:2
"""

X = students.drop(["Target"],axis =1)
y = students["Target"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 123)

print(f'Total # of sample in whole dataset: {len(X)}')
print(f'Total # of sample in train dataset: {len(X_train)}')
print(f'Total # of sample in test dataset: {len(X_test)}')

"""#### Standarization"""

# Identify numerical features
numerical_features = X_train.select_dtypes(include=['int64', 'float64']).columns

# Scale numerical features using StandardScaler
scaler = StandardScaler()
X_train[numerical_features] = scaler.fit_transform(X_train[numerical_features])
X_test[numerical_features] = scaler.transform(X_test[numerical_features])
display(X_train[numerical_features])

X_train[numerical_features].describe().round(4)

"""### Model Development

Kita akan membuat tiga buah model machine learning dangan algoritma berikut:

- Gradient Boosting (Dalam Hal ini kita bakalan pake XGBoost)
- Logistic Regression
- SVM
- KNN
- Random Forest

**Gradient Boosting (GBM)**  
Gradient Boosting bekerja dengan membangun model secara bertahap, di mana setiap model baru mencoba memperbaiki kesalahan model sebelumnya. Model ini menggunakan pendekatan boosting, yaitu menggabungkan beberapa pohon keputusan lemah (weak learners) menjadi model yang lebih kuat. GBM sangat efektif dalam menangani data tabular dan sering digunakan dalam kompetisi machine learning karena kemampuannya dalam menangkap pola yang kompleks.
"""

# Gradient Boosting (XGBoost alternative)
gb = GradientBoostingClassifier(random_state=123)  # You can tune hyperparameters
gb.fit(X_train, y_train)
gb_predictions = gb.predict(X_test)

"""**Logistic Regression**  
Logistic Regression adalah model statistik yang digunakan untuk memprediksi probabilitas suatu peristiwa berdasarkan variabel input. Model ini bekerja dengan menggunakan fungsi sigmoid untuk mengubah hasil regresi linear menjadi nilai probabilitas antara 0 dan 1. Logistic Regression cocok digunakan dalam kasus klasifikasi biner seperti memprediksi apakah seorang siswa akan lulus atau tidak.
"""

# Logistic Regression
logreg = LogisticRegression(max_iter=1000, random_state=123)
logreg.fit(X_train, y_train)
logreg_predictions = logreg.predict(X_test)

"""**Support Vector Machine (SVM)**  
SVM bekerja dengan mencari hyperplane terbaik yang dapat memisahkan kelas-kelas dalam data. Model ini menggunakan konsep margin maksimal, yaitu memilih hyperplane yang memiliki jarak terjauh dari titik-titik terdekat dari setiap kelas. Jika data tidak dapat dipisahkan secara linear, SVM dapat menggunakan kernel trick untuk mengubah data ke dimensi yang lebih tinggi agar lebih mudah dipisahkan. SVM sangat efektif dalam menangani data yang tidak terstruktur dan memiliki distribusi yang kompleks.
"""

# Support Vector Machine (SVM)
svm = SVC(kernel="rbf", random_state=123)
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

"""**Random Forest**

Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Apa itu model ensemble? Sederhananya, ia merupakan model prediksi yang terdiri dari beberapa model dan bekerja secara bersama-sama.
"""

# buat model prediksi
RF = RandomForestClassifier(n_estimators= 474, max_depth= 20, random_state=55, n_jobs=-1)
RF.fit(X_train, y_train)
RF_predictions = RF.predict(X_test)

"""### Evaluation"""

# Lakukan scaling terhadap fitur numerik pada X_test sehingga memiliki rata-rata=0 dan varians=1
X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

# Evaluate using classification metrics
# Buat variabel scores yang isinya adalah dataframe nilai accuracy, precision, recall, f1 score data train dan test pada masing-masing algoritma
scores = pd.DataFrame(columns=['train_accuracy', 'test_accuracy', 'train_precision', 'test_precision', 'train_recall', 'test_recall', 'train_f1', 'test_f1'], index=['GB','LG','SVM'])

# Buat dictionary untuk setiap algoritma yang digunakan
model_dict = {'GB': gb, 'LG': logreg, 'SVM': svm, 'RF': RF}

# Hitung metrik evaluasi masing-masing algoritma pada data train dan test
for name, model in model_dict.items():
    scores.loc[name, 'train_accuracy'] = accuracy_score(y_true=y_train, y_pred=model.predict(X_train))
    scores.loc[name, 'test_accuracy'] = accuracy_score(y_true=y_test, y_pred=model.predict(X_test))
    scores.loc[name, 'train_precision'] = precision_score(y_true=y_train, y_pred=model.predict(X_train), average='weighted')
    scores.loc[name, 'test_precision'] = precision_score(y_true=y_test, y_pred=model.predict(X_test), average='weighted')
    scores.loc[name, 'train_recall'] = recall_score(y_true=y_train, y_pred=model.predict(X_train), average='weighted')
    scores.loc[name, 'test_recall'] = recall_score(y_true=y_test, y_pred=model.predict(X_test), average='weighted')
    scores.loc[name, 'train_f1'] = f1_score(y_true=y_train, y_pred=model.predict(X_train), average='weighted')
    scores.loc[name, 'test_f1'] = f1_score(y_true=y_test, y_pred=model.predict(X_test), average='weighted')

# Print the scores
display(scores)

# Print classification report for each model
for name, model in model_dict.items():
    print(f"Classification Report for {name}:")
    print(classification_report(y_test, model.predict(X_test)))

"""Based on F1 Score, kita bakal coba pake model Gradient Boosting"""

# prompt: Make a direct prediction and classification for the test data based on the already created model, also make a comparison with the real test data on a table format

import pandas as pd
# Create a DataFrame for comparison
comparison_df = pd.DataFrame({'Actual': y_test, 'GB_Prediction': gb_predictions,
                             'LogReg_Prediction': logreg_predictions, 'SVM_Prediction': svm_predictions, 'RF_Prediction': RF_predictions})

# Display the comparison table
comparison_df

"""### Side Quest

Jujur masih penasaran buat modelnya kalo kita bisa ngeput fine-tuning secara maksimum, jadi ak bakalan coba untuk Test Fine-Tuning dulu pake Optuna buat dapet Parameter yang pas buat Trainingnyaa
"""

def objective_rf(trial):
    n_estimators = trial.suggest_int("n_estimators", 100, 1000)
    max_depth = trial.suggest_int("max_depth", 5, 30)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, min_samples_split=min_samples_split, random_state=123, n_jobs=-1
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return 1 - score

def objective_gb(trial):
    n_estimators = trial.suggest_int("n_estimators", 50, 500)
    learning_rate = trial.suggest_float("learning_rate", 0.01, 0.3)
    max_depth = trial.suggest_int("max_depth", 3, 10)

    model = GradientBoostingClassifier(
        n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth, random_state=123
    )
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return 1 - score

def objective_lr(trial):
    C = trial.suggest_float("C", 0.1, 10)
    penalty = trial.suggest_categorical("penalty", ["l1", "l2"])

    model = LogisticRegression(C=C, penalty=penalty, random_state=123, max_iter=1000)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return 1 - score

def objective_svm(trial):
    C = trial.suggest_float("C", 0.1, 10)
    kernel = trial.suggest_categorical("kernel", ["linear", "rbf", "poly"])
    gamma = trial.suggest_categorical("gamma", ["scale", "auto"])

    model = SVC(C=C, kernel=kernel, gamma=gamma, random_state=123)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="accuracy").mean()
    return 1 - score

# Optimize Random Forest
study_rf = optuna.create_study(direction='minimize')
study_rf.optimize(objective_rf, n_trials=50)
print("Best parameters for Random Forest:", study_rf.best_params)

# Optimize Gradient Boosting
study_gb = optuna.create_study(direction='minimize')
study_gb.optimize(objective_gb, n_trials=50)
print("Best parameters for Gradient Boosting:", study_gb.best_params)

# Optimize Logistic Regression
study_lr = optuna.create_study(direction='minimize')
study_lr.optimize(objective_lr, n_trials=50)
print("Best parameters for Logistic Regression:", study_lr.best_params)

# Optimize SVM
study_svm = optuna.create_study(direction='minimize')
study_svm.optimize(objective_svm, n_trials=50)
print("Best parameters for SVM:", study_svm.best_params)

