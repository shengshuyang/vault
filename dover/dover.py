# # %% loading data
# import urllib.request

# url = 'http://storage.googleapis.com/dover-ml-pub/data-science-challenge-data.json'
# response = urllib.request.urlopen(url)

# %% json parsing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

data = []

def encode_employment_status(row):
    current_years = None
    if 'positions' in row:
        curr_pos_list = list(filter(lambda x: x['end_date'] is None, row['positions']))
        if curr_pos_list:
            curr_pos = curr_pos_list[0]
            current_years = curr_pos['current_years_at_company']
    if current_years is None:
        return 'unemployed'
    elif current_years < 1.0:
        return 'less_than_1'
    elif current_years < 3:
        return '1to3'
    elif current_years < 5:
        return '3to5'
    else:
        return '5plus'

def encode_company(val):
    if val == 10000:
        return 'unknown'
    else:
        return str(val // 4 + 1)

def encode_school(val):
    if val == 10000:
        return 'unknown'
    else:
        return str(val // 2 + 1)

# TODO F_Industry isn't very usable
with open('data-science-challenge-data.json') as f:
    for jsonObj in f:
        row = json.loads(jsonObj)
        for key in row['features']['f_scores']:
            row[key] = row['features']['f_scores'][key]
        for key in row['features']:
            row[key] = row['features'][key]
        for col in ['id', 'person_id', 'features', 'date_contacted', 'date_responded', 'job']:
            del row[col]
        row['status'] = encode_employment_status(row)
        for key in ['educations', 'F_Industry', 'f_scores', 'positions', 'currently_employed']:
            if key in row:
                del row[key]
        row['F_Co'] = encode_company(row['F_Co'])
        row['F_School'] = encode_school(row['F_School'])
        row['F_GradSchool'] = encode_school(row['F_GradSchool'])
        data.append(row)

# %% fill nulls
X = pd.DataFrame(data)
X["intersted_in_role"] = X["intersted_in_role"].fillna(False).astype(int)
X["F_CompanySize"] = X["F_CompanySize"].fillna(0.5).astype(float)

X.dtypes

# %% feature stats
for col in X:
    print(X[col].value_counts(dropna=False))
    print('-' * 20)

# %% feature correlations before encoding
import seaborn as sns
corr = X.corr()
sns.heatmap(corr, xticklabels=True, yticklabels=True)

# %% one hot encoding
categorical_features = ['F_Co', 'F_Major', 'F_GradSchool', 'F_School', 'status']
for col in categorical_features:
    encoded_features = pd.get_dummies(X[col], prefix=col)
    X = pd.concat([X, encoded_features], axis=1)
    del X[col]
# %% feature correlations after encoding
import seaborn as sns
corr = X.corr()
sns.heatmap(corr, xticklabels=True, yticklabels=True)

# %% stripping out the labels
y = X['intersted_in_role']
del X['intersted_in_role']

# %% get column names
cols = X.columns.values.tolist()

# %% convert to np array
y = y.to_numpy(dtype=float)
X = X.to_numpy(dtype=float)

# %% train test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, stratify=y)

# %% training a simple svm classifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    # ('svc', LinearSVC(C=1.0e-2, penalty='l1', dual=False, loss='squared_hinge', class_weight='balanced')),
    ('svc', LogisticRegression(C=1.0e-2, penalty='l1', solver='liblinear', dual=False, class_weight='balanced')),
])

pipe.fit(X_train, y_train)

# %% scoring
y_test_pred = pipe.predict(X_test)
y_test_prob = pipe.predict_proba(X_test)

plt.hist(y_test_prob[y_test == 1][:, 1], bins=30, alpha=1.0, label='positive', log=True, density=True)
plt.hist(y_test_prob[y_test == 0][:, 1], bins=30, alpha=0.5, label='negative', log=True, density=True)
plt.legend(loc='upper right')
plt.title('Score histogram by class')
plt.show()
# %% plot auc roc
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.metrics import roc_curve

fpr, tpr, _ = roc_curve(y_test, y_test_pred)
auc_test = roc_auc_score(y_test, y_test_pred)
acc_test = accuracy_score(y_test, y_test_pred)
f1_score = f1_score(y_test, y_test_pred)
plt.figure()
plt.plot(fpr, tpr, color='darkorange')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title(f'ROC curve (auc = {auc_test}, accuracy = {acc_test}, f1_score = {f1_score})')

plt.show()

# %% sort column names by coefficient values
coeffs = pipe.named_steps['svc'].coef_.tolist()[0]
zipped = list(zip(cols, coeffs))
zipped.sort(key=lambda x: x[1], reverse=True)
cols = [col for col, _ in zipped]
coeffs = [coeff for _, coeff in zipped]

# %% plot coefficients
plt.rcdefaults()
fig, ax = plt.subplots()

y_pos = np.arange(len(coeffs))

ax.barh(y_pos, coeffs, align='center')
ax.set_yticks(y_pos)
ax.set_yticklabels(cols, va='bottom', fontsize=6)
ax.tick_params(axis='y', pad=4)
ax.grid()
ax.invert_yaxis()  # labels read top-to-bottom
ax.set_xlabel('coefficient')
ax.set_title('Which features are more important?')
plt.show()
