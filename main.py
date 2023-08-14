from sklearn import datasets
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score

from AutoGluon.auto_gluon import AutoGluonEstimator

# Prepare dataset and train-val split

dataset_dict = datasets.fetch_california_housing(as_frame=True)

X_data = dataset_dict['data']
y_data = dataset_dict['target']

# train-val split
val_fraction = 0.2

X_train = X_data.iloc[:int((1-val_fraction)*X_data.shape[0]), :]
y_train = y_data.iloc[:int((1-val_fraction)*X_data.shape[0])]
X_val = X_data.iloc[int((1-val_fraction)*X_data.shape[0]):, :]
y_val = y_data.iloc[int((1-val_fraction)*X_data.shape[0]):]

# AutoML framework selection and training

params = dict(
label = y_train.name
)

frameworks_selection = {
    'AutoGluon': AutoGluonEstimator(**params),
    #'AutoSklearn': AutoSklearnEstimator(),
    #'TPOT': TPOTEstimator()
}

model_selected = 'AutoGluon'

model = frameworks_selection[model_selected]

model.fit(X_train, y_train)

# Basic model results evaluation

# Prediction on validation set
pred_val = model.predict(X_val)

metrics = {
    'MAE': mean_absolute_error(y_val, pred_val),
    'MSE': mean_squared_error(y_val, pred_val),
    'MAPE': mean_absolute_percentage_error(y_val, pred_val),
    'R2': r2_score(y_val, pred_val)
}

print('Obtained metrics values for validation set: \n')
for metric, value in metrics.items():
  print(metric, value)