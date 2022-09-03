########## DİABETES PREDİCTİONS WİTH LOGİSTİC REGRESSİON ###########

# İS PROBLEMI
 # ozellikleri belirtilen kisilerin diyabet hastası olup olmadıklarını tahmın edebılecek
 # bir makine ogrenmesı modelı gelıstırebılır mısınız?

# Veri seti ABD'deki Ulusal Diyabet-Sindirim-Böbrek Hastalıkları Enstitüleri'nde tutulan büyük veri setinin
# parçasıdır. ABD'deki Arizona Eyaleti'nin en büyük 5. şehri olan Phoenix şehrinde yaşayan 21 yaş ve üzerinde olan
# Pima Indian kadınları üzerinde yapılan diyabet araştırması için kullanılan verilerdir. 768 gözlem ve 8 sayısal
# bağımsız değişkenden oluşmaktadır. Hedef değişken "outcome" olarak belirtilmiş olup; 1 diyabet test sonucunun
# pozitif oluşunu, 0 ise negatif oluşunu belirtmektedir.

# Pregnancies: Hamilelik sayısı
# Glucose: Glikoz
# BloodPressure: Kan basıncı (Diastolic(Küçük Tansiyon))
# SkinThickness: Cilt Kalınlığı
# Insulin: İnsülin.
# BMI: Beden kitle indeksi.
# DiabetesPedigreeFunction: Soyumuzdaki kişilere göre diyabet olma ihtimalimizi hesaplayan bir fonksiyon.
# Age: Yaş (yıl)
# Outcome: Kişinin diyabet olup olmadığı bilgisi. Hastalığa sahip (1) ya da değil (0)

# 1. exploratory data analysis
# 2. data preprocessing
# 3. model & prediction
# 4. model evulation
# 5. model validations:holdout
# 6. model validations : 10 ford cross validations
# 7. predictions for a  new observations


# Gerekli Kütüphane ve Fonksiyonlar
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, plot_roc_curve, roc_auc_score
import warnings
warnings.simplefilter(action="ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def replace_with_thresholds(dataframe, variable, q1=0.05, q3=0.95):
    low_limit, up_limit = outlier_thresholds(dataframe, variable, q1=0.05, q3=0.95)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

df= pd.read_csv("datasets/diabetes.csv")
df.head()
df.shape

# target analizi
df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

df["Outcome"].value_counts() * 100 / len(df) #yuzde olarak gormke ıcın

df.describe().T

df["Glucose"].hist(bins=20)
plt.xlabel("Glucose")
plt.show()

def plot_numerical_col(dataframe,numerical_col):
    dataframe[numerical_col].hist(bins=20)
    plt.xlabel(numerical_col)
    plt.show(block=True)
for col in df.columns:
    plot_numerical_col(df, col)
# outcome harıcı olanları ıstedık
cols=[col for col in df.columns if "Outcome" not in col]
for col in df.columns:
    plot_numerical_col(df, col)

# target and features
df.groupby("Outcome").agg({"Pregnancies":"mean"})

def target_summary_with_num(dataframe,target,numerical_num):
    print(dataframe.groupby(target).agg({numerical_num: "mean"}), end="\n\n\n")

for col in df.columns:
    target_summary_with_num(df, "Outcome", col)

df.isnull().sum()
df.describe().T

for col in cols:
    print(col, check_outlier(df, col)) # her columnda aykırı deger varmı dıye bakyoruz

replace_with_thresholds(df, "Insulin")

for col in cols:
    df[col] = RobustScaler().fit_transform(df[[col]])

df.head()

# MODEL AND PREDİCTİONS

y = df["Outcome"]
X = df.drop(["Outcome"], axis=1)

log_model = LogisticRegression().fit(X, y) # model kuruldu
log_model.intercept_
# [-1.23439588] bu deger b yanı bıas degerı
log_model.coef_

y_pred = log_model.predict(X)
y_pred[0:10]
y[0:10]

# MODEL EVULATION
# karmasıklık matrısı gosterımı ıcın  bır fonksıyon
def plot_confusion_matrix(y,y_pred):
    acc= round(accuracy_score(y, y_pred), 2)
    cm = confusion_matrix(y, y_pred)
    sns.heatmap(cm, annot=True, fmt=".0f")
    plt.xlabel("y_pred")
    plt.ylabel("y")
    plt.title("accuracy score: {0}".format(acc), size=10)
    plt.show()

plot_confusion_matrix(y,y_pred)

print(classification_report(y, y_pred)) # cıktıdakı 1 degerne bakarız 1 degerıne bakarak yorum yapabılrız

# accuracy:0.78
# precision:0.74
# recall:0.58
# f1 score:0.65

# roc egrısınq gore yapalım
y_prob = log_model.predict_proba(X)[:, 1]
roc_auc_score(y, y_prob)
# 0.8393955223880598


# MODEL DOGRULAMA  MODEL VALIDATİON :HOLDOUT
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17)

log_model= LogisticRegression().fit(X_train,y_train)
y_pred=log_model.predict(X_test)
y_prob= log_model.predict_proba(X_test)[:,1]
print(classification_report(y_test,y_pred))


## 10 KATLI CAPRAZ DOGRULAMA CROSS VALIDATION
y= df["Outcome"]
X= df.drop(["Outcome"], axis=1)
log_model = LogisticRegression().fit(X, y) # az sayıda verımız varsa tum verı setını kullanarak da cross val yapabılırız
cv_results = cross_validate(log_model,
                           X,y,
                           cv=5,
                           scoring=["accuracy","precision","recall","f1","roc_auc"])
cv_results["test_accuracy"].mean()

cv_results['test_precision'].mean()

cv_results['test_recall'].mean()

cv_results['test_f1'].mean()

cv_results['test_roc_auc'].mean()

random_user = X.sample(1, random_state=45)
log_model.predict(random_user)





