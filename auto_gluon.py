from autogluon.tabular import TabularDataset, TabularPredictor
import pandas as pd

class AutoGluonEstimator():
    def __init__(self, problem_type='Regressor', label='', eval_metric='RMSE', time_limit=60,
                 model_path='model/', task='CPU', num_workers=-1):
        self.name = "AutoGluon"+problem_type
        self.type = problem_type
        self.label = label
        self.eval_metric = eval_metric
        self.time_limit = time_limit
        self.model_path = model_path
        self.task = task
        self.num_workers = num_workers
        self.model = None

        return self

    def fit(self, X_train, y_train, X_val, y_val):
        if self.problem_type=='Regressor':
            problem = 'regression'
        elif self.problem_type=='Classifier':
            problem = 'binary'
        else:
            raise ValueError('Other problem types not supported here!')

        train_data = TabularDataset(pd.concat([X_train, y_train], axis=1))
        val_data = TabularDataset(pd.concat([X_val, y_val], axis=1))

        self.model = TabularPredictor(label=self.label, problem_type=problem, eval_metric=self.eval_metric,
                                      path=self.model_path)
        self.model.fit(train_data, tuning_data=val_data, time_limit=self.time_limit)

        return self

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)

    def get_model(self):
        return self.model


#if __name__ == 'main':
    # pd.read_csv
    # fit
    # predict