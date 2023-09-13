import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')
from copy import deepcopy
from scipy.signal import periodogram
from statsmodels.tsa.stattools import pacf, acf
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm_notebook

class Forecaster:

    def __init__(self, df, x, y, group_features = [], categorical_features = [], scoring_metric = metrics.mean_absolute_percentage_error, keep_id = False):
        self.df = df

        self.keep_id = keep_id

        self.date = x
        self.target = y
        self.group_features = group_features.copy()
        self.categorical_features = categorical_features.copy()
        self.categorical_features += ['dayofweek']
        self.scoring_metric = scoring_metric

        self.df[self.date] = pd.to_datetime(self.df[self.date])
        self.time_delta = pd.Timedelta(self.df[self.date].unique()[1] - self.df[self.date].unique()[0])

    def make_future_dataframe(self, periods, fill_zero = []):

        future = pd.DataFrame()

        start_date = self.df[self.date].iloc[-1] + self.time_delta
        end_date = start_date + (periods * self.time_delta)
        date_range = pd.date_range(start_date, end_date, freq = self.time_delta)

        to_copy = self.df[self.df[self.date] == self.df[self.date].iloc[-1]].copy()

        for date in date_range:
            to_copy[self.date] = date

            # to_copy[to_copy.columns.drop([self.id_, self.date, 'date_index', *self.group_features, *self.categorical_features])] = 0
            to_copy[fill_zero] = 0
            to_copy[self.target] = np.nan

            future = pd.concat([future, to_copy])

        self.df = pd.concat([self.df, future]).reset_index(drop = True)

    def create_seasonality(self, data):

        ret = pd.DataFrame()

        fs = pd.Timedelta(365, 'd')/self.time_delta

        # freqencies, spectrum = periodogram(
        #     data[data[self.target].notna()].groupby(self.date)[self.target].mean(),
        #     fs=fs,
        #     detrend='linear',
        #     window="boxcar",
        #     scaling='spectrum',
        # )
        
        freqs = np.array([1, 2, 4, 6, 12, 26, 52, 104, 159, 365, 730, 8760])
        names = ['Annual', 'Semiannual', 'Quarterly', 'Bimonthly', 'Monthly', 'Biweekly', 'Weekly',
                 'Semiweekly', '2.3_daily', 'Daily', 'Semidaily', 'Hourly']

        Seasonality = dict(zip(names, np.round(fs / freqs, 1)))

        for s in Seasonality:
            if Seasonality[s]>1:
                ret[f'sin_{s}'] = np.sin(data['date_index'] * (2*np.pi / Seasonality[s]))
                ret[f'cos_{s}'] = np.cos(data['date_index'] * (2*np.pi / Seasonality[s]))
        return pd.concat([data, ret], axis = 1)

    def create_lags(self, data, lags = True):

        ret = pd.DataFrame()

        if lags == True:
            average_sales = data[data[self.target].notna()].groupby(self.date)[self.target].mean()
            plt.show()
            alpha=0.05
            method="ywm"
            nlags = len(average_sales)//2-1
            lags, thresh = pacf(average_sales, nlags=nlags, alpha=alpha)#, method=method)
            lags = np.where(((lags > thresh[:, 1] - lags)) | ((lags < thresh[:, 0] - lags)))[0][1:]

        if len(lags):

            if len(self.group_features):
                for lag in lags:
                    ret[f'lag_{lag}'] = data.groupby(self.group_features)[self.target].transform(lambda x: x.shift(lag))

            elif len(self.group_features) == 0:
                for lag in lags:
                    ret[f'lag_{lag}'] = data[self.target].shift(lag)

            ret = pd.concat([data, ret], axis = 1)
            ret.index.unique()[lags[-1]]
            nan_head = ret.index.unique()[lags[-1]]

            ret = ret[nan_head:]

            return ret, lags

        return data, lags

    def train_valid_split(self, data, split = 0.7):

        dates = data.index.unique()
        n = len(dates)
        split_date = dates[int(n*split)]

        X_valid = data[data[self.target].notna()]

        y_valid = X_valid.loc[split_date:].sort_values(by = [self.date, self.id_])[self.target].copy()
        X_valid.loc[split_date:, self.target] = np.nan

        return X_valid, y_valid

    def validate(
        self,
        model,
        seasonality = False, 
        lag = False, 
        by = None, 
        plot = False, 
        fit_kwargs = {}
    ):

        data = self.df[self.df[self.target].notna()].copy()

        if not self.keep_id:
            self.id_ = 'id'
            data['id'] = 0

        if self.keep_id:
            self.id_ = self.keep_id
        data['hour'] = data[self.date].dt.hour
        data['year'] = data[self.date].dt.year
        data['month'] = data[self.date].dt.month
        data['dayofweek'] = data[self.date].dt.dayofweek
        data['dayofmonth'] = data[self.date].dt.day
        data["dayofyear"] = data[self.date].dt.dayofyear
        data["weekofyear"] = data[self.date].dt.isocalendar().week.astype(int)
        data['date_index'] = data[self.date].factorize()[0]
        
        try:
            model_name = str(list(model.layers)[0]).split('.')[-1].split('object')[0]
        except:
            model_name = str(model).split("(")[0]

        if len(self.group_features) == 0:
            df_valid , y_valid = self.train_valid_split(data.set_index(self.date))

            if seasonality:
                df_valid = self.create_seasonality(df_valid)

            if lag == False:

                dummy_vars = self.categorical_features.copy()
                label_encode = []
                for col in dummy_vars:
                    if len(df_valid[col].unique()) > 500:
                        encoder = LabelEncoder()
                        df_valid[col] = encoder.fit_transform(df_valid[col])
                        label_encode = []

                for col in label_encode:
                    dummy_vars.pop(dummy_vars.index(col))

                df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                Xscaler = StandardScaler()
                yscaler = StandardScaler()

                df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                y_train = df_valid[df_valid[self.target].notna()][self.target]
                X_valid = df_valid[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])
                valid_index = X_valid.index

                model_ = deepcopy(model)
                
                if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                    X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                if 'Conv1D' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                    X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1))                

                model_.fit(X_train, y_train, **fit_kwargs)
                
                try:
                    y_pred = model_.predict(X_valid, verbose = 0)
                    y_pred = y_pred.flatten()   
                except:
                    y_pred = model_.predict(X_valid).flatten()

                y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)

                y_pred = pd.Series(y_pred, index = valid_index)

                try:
                    y_train_pred = model_.predict(X_train, verbose = 0).flatten()
                except:
                    y_train_pred = model_.predict(X_train).flatten()

                y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)

                training_score = self.scoring_metric(y_train, y_train_pred)
                test_score = self.scoring_metric(y_valid, y_pred)

                scores = pd.DataFrame({model_name: [training_score, test_score]}, index = ['Training score', 'Test score']).T
                display(scores)

                if plot == True:
                    ax = y_train.plot(figsize = (10, 3))
                    y_valid.plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                    y_pred.plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.keep_id:
                    y_pred_w_id = pd.DataFrame({self.id_: X_valid[id_], self.target: y_pred}, index = y_pred.index)
                    return y_pred_w_id, scores

                return y_pred.reset_index(), scores

            if lag != False:

                df_valid, lags = self.create_lags(df_valid, lag)

                dummy_vars = self.categorical_features.copy()
                label_encode = []
                for col in dummy_vars:
                    if len(df_valid[col].unique()) > 500:
                        encoder = LabelEncoder()
                        df_valid[col] = encoder.fit_transform(df_valid[col])
                        label_encode = []

                for col in label_encode:
                    dummy_vars.pop(dummy_vars.index(col))

                df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                Xscaler = StandardScaler()
                yscaler = StandardScaler()

                df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

                if 'Conv1D' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                
                model_ = deepcopy(model)
                model_.fit(X_train, y_train, **fit_kwargs)

                try:
                    y_train_pred = model_.predict(X_train, verbose = 0).flatten()
                except:
                    y_train_pred = model_.predict(X_train).flatten()

                y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                training_score = self.scoring_metric(y_train, y_train_pred)

                forecast_index = df_valid[df_valid[self.target].isna()].index
                for i in tqdm_notebook(range(len(forecast_index))):

                    future_index = df_valid[df_valid[self.target].isna()].index[0]
                    future_X = df_valid.loc[future_index:future_index].drop(columns = [self.target, self.id_])

                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        future_X = np.reshape(future_X.values, (future_X.shape[0], 1, future_X.shape[1]))

                    if 'Conv1D' in str(model_name):
                        future_X = np.reshape(future_X.values, (future_X.shape[0], future_X.shape[1], 1))

                    try:
                        future_1 = model_.predict(future_X, verbose = 0).flatten()[0]
                    except:
                        future_1 = model_.predict(future_X).flatten()[0]

                    df_valid.loc[future_index, self.target] = future_1

                    try:
                        unique_dates = pd.Series(df_valid.index.unique())
                        unique_index = np.where(unique_dates == future_index)[0][0]
                        for lag_ in lags:
                            df_valid.loc[unique_dates[unique_index + lag_], f'lag_{lag_}'] = future_1
                    except:
                        pass

                y_pred_w_id = df_valid.loc[forecast_index, [self.id_, self.target]].copy()
                y_pred_w_id[self.target] = yscaler.inverse_transform(y_pred_w_id[self.target].values.reshape(-1, 1)).flatten()

                y_pred = y_pred_w_id[self.target]

                test_score = self.scoring_metric(y_valid, y_pred)

                scores = pd.DataFrame({model_name: [training_score, test_score]}, index = ['Training score', 'Test score']).T
                display(scores)

                if plot == True:
                    ax = y_train.plot(figsize = (10, 3))
                    y_valid.plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                    y_pred.plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.keep_id:
                    return y_pred_w_id, scores

                return y_pred.reset_index(), scores

        if by == None and len(self.group_features):
            df_valid , y_valid = self.train_valid_split(data.set_index(self.date))

            if seasonality:
                df_valid = self.create_seasonality(df_valid)

            if lag == False:

                dummy_vars = (self.group_features + self.categorical_features).copy()
                label_encode = []
                for col in dummy_vars:
                    if len(df_valid[col].unique()) > 500:
                        encoder = LabelEncoder()
                        df_valid[col] = encoder.fit_transform(df_valid[col])
                        label_encode = []

                for col in label_encode:
                    dummy_vars.pop(dummy_vars.index(col))

                df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                Xscaler = StandardScaler()
                yscaler = StandardScaler()

                df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                X_valid = df_valid[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])
                valid_index = X_valid.index
                if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                    X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                if 'Conv1D' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                    X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1))            
                model_ = deepcopy(model)
                model_.fit(X_train, y_train, **fit_kwargs)

                try:
                    y_pred = model_.predict(X_valid, verbose = 0).flatten()
                except:
                    y_pred = model_.predict(X_valid).flatten()
                

                y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)


                y_pred_w_id = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred}, index = valid_index)
                y_pred = y_pred_w_id.sort_values(by = [self.date, self.id_])[self.target]

                try:
                    y_train_pred = model_.predict(X_train, verbose = 0).flatten()
                except:
                    y_train_pred = model_.predict(X_train).flatten()
                
                training_score = self.scoring_metric(y_train, y_train_pred)
                test_score = self.scoring_metric(y_valid, y_pred)

                scores = pd.DataFrame({model_name: [training_score, test_score]}, index = ['Training score', 'Test score']).T
                display(scores)

                if plot == True:
                    ax = y_train.reset_index().groupby(self.date).mean().plot(figsize = (10, 3))
                    y_valid.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                    y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.keep_id:
                    return y_pred_w_id, scores

                return y_pred.reset_index(), scores

            if lag != False:

                df_valid, lags = self.create_lags(df_valid, lag)

                dummy_vars = (self.group_features + self.categorical_features).copy()
                label_encode = []
                for col in dummy_vars:
                    if len(df_valid[col].unique()) > 500:
                        encoder = LabelEncoder()
                        df_valid[col] = encoder.fit_transform(df_valid[col])
                        label_encode = []

                for col in label_encode:
                    dummy_vars.pop(dummy_vars.index(col))

                df_valid_dummy = pd.get_dummies(df_valid, columns = dummy_vars)
                for feature in self.group_features:
                    if feature in dummy_vars:
                        df_valid_dummy[feature] = df_valid[feature]

                df_valid = df_valid_dummy
                del df_valid_dummy

                Xscaler = StandardScaler()
                yscaler = StandardScaler()

                df_valid[df_valid.columns.drop([self.id_, self.target, *self.group_features])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target, *self.group_features])])
                df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]

                if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

                if 'Conv1D' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                    
                model_ = deepcopy(model)

                model_.fit(X_train, y_train, **fit_kwargs)

                try:
                    y_train_pred = model_.predict(X_train, verbose = 0).flatten()
                except:
                    y_train_pred = model_.predict(X_train).flatten()

                y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                training_score = self.scoring_metric(y_train, y_train_pred)

                y_pred_w_ids = []

                for group in tqdm_notebook(df_valid.groupby(self.group_features)):

                    df_group = group[1]

                    forecast_index = df_group[df_group[self.target].isna()].index
                    for i in range(len(forecast_index)):

                        future_index = df_group[df_group[self.target].isna()].index[0]
                        future_X = df_group.loc[future_index:future_index].drop(columns = [self.target, self.id_, *self.group_features])
                        if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                            future_X = np.reshape(future_X, (future_X.shape[0], 1, future_X.shape[1]))

                        if 'Conv1D' in str(model_name):
                            future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))
                        try:
                            future_1 = model_.predict(future_X, verbose = 0).flatten()[0]
                        except:
                            future_1 = model_.predict(future_X).flatten()[0]

                        df_group.loc[future_index, self.target] = future_1

                        try:
                            unique_dates = pd.Series(df_group.index.unique())
                            unique_index = np.where(unique_dates == future_index)[0][0]
                            for lag_ in lags:
                                df_group.loc[unique_dates[unique_index + lag_], f'lag_{lag_}'] = future_1                        
                        except:
                            pass

                    y_pred_w_ids.append(df_group.loc[forecast_index, [self.id_, self.target]].copy())

                y_pred_w_id = pd.concat(y_pred_w_ids).sort_values(by = [self.date, self.id_])

                y_pred_w_id[self.target] = yscaler.inverse_transform(y_pred_w_id[self.target].values.reshape(-1, 1)).flatten()

                y_pred = y_pred_w_id[self.target]

                test_score = self.scoring_metric(y_valid, y_pred)

                scores = pd.DataFrame({model_name: [training_score, test_score]}, index = ['Training score', 'Test score']).T
                display(scores)

                if plot == True:
                    ax = y_train.reset_index().groupby(self.date).mean().plot(figsize = (10, 3))
                    y_valid.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                    y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.keep_id:
                    return y_pred_w_id, scores

                return y_pred.reset_index(), scores

        if len(by) == len(self.group_features):

            models = {}
            training_scores = {}
            test_scores = {}
            y_pred_w_ids = []
            y_preds = []
            y_trains = []
            y_valids = []

            for group in tqdm_notebook(data.groupby(self.group_features)):

                df_valid , y_valid = self.train_valid_split(group[1][group[1][self.target].notna()].set_index(self.date))
                y_valids.append(y_valid)

                if seasonality:
                    df_valid = self.create_seasonality(df_valid)

                if lag == False:

                    dummy_vars = (self.categorical_features).copy()
                    label_encode = []
                    for col in dummy_vars:
                        if len(df_valid[col].unique()) > 500:
                            encoder = LabelEncoder()
                            df_valid[col] = encoder.fit_transform(df_valid[col])
                            label_encode = []

                    for col in label_encode:
                        dummy_vars.pop(dummy_vars.index(col))

                    df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                    Xscaler = StandardScaler()
                    yscaler = StandardScaler()

                    df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                    df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]

                    X_valid = df_valid.drop(columns = self.group_features)[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])
                    valid_index = X_valid.index
                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                    if 'Conv1D' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1)) 

                    models[group[0]] = deepcopy(model)

                    models[group[0]].fit(X_train, y_train, **fit_kwargs)

                    try:
                        y_pred_ = models[group[0]].predict(X_valid, verbose = 0).flatten()
                    except:
                        y_pred_ = models[group[0]].predict(X_valid).flatten()

                    y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                    y_trains.append(y_train)

                    y_pred_w_id_ = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred_}, index = valid_index)
                    y_pred_w_ids.append(y_pred_w_id_)
                    y_pred_ = y_pred_w_id_.sort_values(by = [self.date, self.id_])[self.target]
                    y_preds.append(y_pred_)

                    try:
                        y_train_pred = models[group[0]].predict(X_train, verbose = 0).flatten()
                    except:
                        y_train_pred = models[group[0]].predict(X_train).flatten()

                    training_scores[group[0]] = (self.scoring_metric(y_train, y_train_pred))
                    test_scores[group[0]] = (self.scoring_metric(y_valid, y_pred_))

                if lag != False:

                    df_valid, lags = self.create_lags(df_valid, lag)

                    dummy_vars = (self.categorical_features).copy()
                    label_encode = []
                    for col in dummy_vars:
                        if len(df_valid[col].unique()) > 500:
                            encoder = LabelEncoder()
                            df_valid[col] = encoder.fit_transform(df_valid[col])
                            label_encode = []

                    for col in label_encode:
                        dummy_vars.pop(dummy_vars.index(col))

                    df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                    Xscaler = StandardScaler()
                    yscaler = StandardScaler()

                    df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                    df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]

                    X_valid = df_valid.drop(columns = self.group_features)[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])

                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                    if 'Conv1D' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1)) 

                    models[group[0]] = deepcopy(model)

                    models[group[0]].fit(X_train, y_train, **fit_kwargs)

                    try:
                        y_train_pred = models[group[0]].predict(X_train, verbose = 0).flatten()
                    except:
                        y_train_pred = models[group[0]].predict(X_train).flatten()


                    y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                    y_trains.append(y_train)
                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                    forecast_index = df_valid[df_valid[self.target].isna()].index
                    for i in (range(len(forecast_index))):

                        future_index = df_valid[df_valid[self.target].isna()].index[0]
                        future_X = df_valid.loc[future_index:future_index].drop(columns = [self.target, self.id_, *self.group_features])
                        if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                            future_X = np.reshape(future_X, (future_X.shape[0], 1, future_X.shape[1]))

                        if 'Conv1D' in str(model_name):
                            future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))
                        try:
                            future_1 = models[group[0]].predict(future_X, verbose = 0).flatten()[0]
                        except:
                            future_1 = models[group[0]].predict(future_X).flatten()[0]

                        df_valid.loc[future_index, self.target] = future_1

                        try:
                            unique_dates = pd.Series(df_valid.index.unique())
                            unique_index = np.where(unique_dates == future_index)[0][0]
                            for lag_ in lags:
                                df_valid.loc[unique_dates[unique_index + lag_], f'lag_{lag_}'] = future_1  
                        except:
                            pass

                    y_pred_w_id_ = df_valid.loc[forecast_index, [self.id_, self.target]].copy()

                    y_pred_w_id_[self.target] = yscaler.inverse_transform(y_pred_w_id_[self.target].values.reshape(-1, 1)).flatten()

                    y_pred_ = y_pred_w_id_[self.target]

                    y_pred_w_ids.append(y_pred_w_id_)
                    y_preds.append(y_pred_)

                    test_scores[group[0]] = self.scoring_metric(y_valid, y_pred_)

            y_pred_w_id = pd.concat(y_pred_w_ids)
            y_pred = pd.concat(y_preds)

            training_score = pd.DataFrame(training_scores, index = ['Training score']).T
            test_score = pd.DataFrame(test_scores, index = ['Test score']).T

            scores = pd.concat([training_score, test_score], axis = 1)

            display(scores)
            display(scores.mean().to_frame().rename(columns = {0: 'AVERAGE'}).T)

            if plot == True:
                ax = pd.concat(y_trains).reset_index().groupby(self.date).mean().plot(figsize = (15, 3))
                pd.concat(y_valids).reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                ax.set_xlabel(self.date)
                ax.set_ylabel(self.target)
                plt.tight_layout()
                plt.show()

            if self.keep_id:
                return y_pred_w_id, scores

            return y_pred.reset_index(), scores

        if len(by) == 1 and len(self.group_features):

            gr_1 = by
            gr_2 = self.group_features.copy()
            gr_2.pop(gr_2.index(by[0]))

            models = {}
            training_scores = {}
            test_scores = {}
            y_pred_w_ids = []
            y_preds = []
            y_trains = []
            y_valids = []

            for group in tqdm_notebook(data.groupby(gr_1)):

                df_valid , y_valid = self.train_valid_split(group[1][group[1][self.target].notna()].set_index(self.date))
                y_valids.append(y_valid)

                if seasonality:
                    df_valid = self.create_seasonality(df_valid)

                if lag == False:

                    dummy_vars = (gr_2 + self.categorical_features).copy()
                    label_encode = []
                    for col in dummy_vars:
                        if len(df_valid[col].unique()) > 500:
                            encoder = LabelEncoder()
                            df_valid[col] = encoder.fit_transform(df_valid[col])
                            label_encode = []

                    for col in label_encode:
                        dummy_vars.pop(dummy_vars.index(col))                   

                    df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                    Xscaler = StandardScaler()
                    yscaler = StandardScaler()

                    df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                    df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                    X_train = df_valid.drop(columns = by)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                    y_train = df_valid.drop(columns = by)[df_valid[self.target].notna()][self.target]

                    X_valid = df_valid.drop(columns = by)[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])
                    valid_index = X_valid.index
                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                    if 'Conv1D' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1)) 

                    models[group[0]] = deepcopy(model)

                    models[group[0]].fit(X_train, y_train, **fit_kwargs)

                    try:
                        y_pred_ = models[group[0]].predict(X_valid, verbose = 0).flatten()
                    except:
                        y_pred_ = models[group[0]].predict(X_valid).flatten()


                    y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                    y_trains.append(y_train)

                    y_pred_w_id_ = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred_}, index = valid_index)
                    y_pred_ = y_pred_w_id_.sort_values(by = [self.date, self.id_])[self.target]

                    y_pred_w_ids.append(y_pred_w_id_)
                    y_preds.append(y_pred_)

                    try:
                        y_train_pred = models[group[0]].predict(X_train, verbose = 0).flatten()
                    except:
                        y_train_pred = models[group[0]].predict(X_train).flatten()

                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)
                    test_scores[group[0]] = self.scoring_metric(y_valid, y_pred_)

                if lag != False:

                    df_valid, lags = self.create_lags(df_valid, lag)

                    dummy_vars = (gr_2 + self.categorical_features).copy()
                    label_encode = []
                    for col in dummy_vars:
                        if len(df_valid[col].unique()) > 500:
                            encoder = LabelEncoder()
                            df_valid[col] = encoder.fit_transform(df_valid[col])
                            label_encode = []

                    for col in label_encode:
                        dummy_vars.pop(dummy_vars.index(col))

                    df_valid_dummy = pd.get_dummies(df_valid, columns = dummy_vars)

                    for feature in gr_2:
                        df_valid_dummy[feature] = df_valid[feature]

                    df_valid = df_valid_dummy
                    del df_valid_dummy

                    Xscaler = StandardScaler()
                    yscaler = StandardScaler()

                    df_valid[df_valid.columns.drop([self.id_, self.target, *gr_2])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target, *gr_2])])
                    df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]

                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

                    if 'Conv1D' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))

                    models[group[0]] = deepcopy(model)

                    models[group[0]].fit(X_train, y_train, **fit_kwargs)

                    try:
                        y_train_pred = models[group[0]].predict(X_train, verbose = 0).flatten()
                    except:
                        y_train_pred = models[group[0]].predict(X_train).flatten()

                    y_pred_w_id[self.target] = yscaler.inverse_transform(y_pred_w_id[self.target].values.reshape(-1, 1)).flatten()
                    y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                    y_trains.append(y_train)

                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                    y_pred_w_ids_ = []

                    for group_ in df_valid.groupby(gr_2):

                        df_group = group_[1]

                        forecast_index = df_group[df_group[self.target].isna()].index
                        for i in range(len(forecast_index)):

                            future_index = df_group[df_group[self.target].isna()].index[0]
                            future_X = df_group.loc[future_index:future_index].drop(columns = [self.target, self.id_, *self.group_features])
                            if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                                future_X = np.reshape(future_X, (future_X.shape[0], 1, future_X.shape[1]))

                            if 'Conv1D' in str(model_name):
                                future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))
                            try:
                                future_1 = models[group[0]].predict(future_X, verbose = 0).flatten()[0]
                            except:
                                future_1 = models[group[0]].predict(future_X).flatten()[0]

                            df_group.loc[future_index, self.target] = future_1

                            try:
                                unique_dates = pd.Series(df_group.index.unique())
                                unique_index = np.where(unique_dates == future_index)[0][0]
                                for lag_ in lags:
                                    df_group.loc[unique_dates[unique_index + lag_], f'lag_{lag_}'] = future_1  
                            except:
                                pass

                        y_pred_w_id__ = df_group.loc[forecast_index, [self.id_, self.target]].copy()
                        y_pred_w_ids_.append(y_pred_w_id__)

                    y_pred_w_id_ = pd.concat(y_pred_w_ids_).sort_values(by = [self.date, self.id_])

                    y_pred_w_id_[self.target] = yscaler.inverse_transform(y_pred_w_id_[self.target].values.reshape(-1, 1)).flatten()

                    y_pred_ = y_pred_w_id_[self.target]

                    y_pred_w_ids.append(y_pred_w_id_)
                    y_preds.append(y_pred_)

                    test_scores[group[0]] = self.scoring_metric(y_valid, y_pred_)

            y_pred_w_id = pd.concat(y_pred_w_ids)
            y_pred = pd.concat(y_preds)

            training_score = pd.DataFrame(training_scores, index = ['Training score']).T
            test_score = pd.DataFrame(test_scores, index = ['Test score']).T

            scores = pd.concat([training_score, test_score], axis = 1)

            display(scores)
            display(scores.mean().to_frame().rename(columns = {0: 'AVERAGE'}).T)

            if plot == True:
                ax = pd.concat(y_trains).reset_index().groupby(self.date).mean().plot(figsize = (15, 3))
                pd.concat(y_valids).reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                ax.set_xlabel(self.date)
                ax.set_ylabel(self.target)
                plt.tight_layout()
                plt.show()

            if self.keep_id:
                return y_pred_w_id, scores

            return y_pred.reset_index(), scores


    def predict(
        self, 
        model, 
        seasonality = False, 
        lag = False, 
        by = None, 
        plot = False, 
        fit_kwargs = {}
    ):

        try:
            model_name = model.layers[0].split('.')[-1].split('object')[0].strip()
        except:
            model_name = str(model).split("(")[0]
        
        data = self.df.copy()

        if not self.keep_id:
            self.id_ = 'id'
            data['id'] = 0

        if self.keep_id:
            self.id_ = self.keep_id
        data['hour'] = data[self.date].dt.hour
        data['year'] = data[self.date].dt.year
        data['month'] = data[self.date].dt.month
        data['dayofweek'] = data[self.date].dt.dayofweek
        data['dayofmonth'] = data[self.date].dt.day
        data["dayofyear"] = data[self.date].dt.dayofyear
        data["weekofyear"] = data[self.date].dt.isocalendar().week.astype(int)
        data['date_index'] = data[self.date].factorize()[0]
        
        try:
            model_name = model.layers[0].split('.')[-1].split('object')[0].strip()
        except:
            model_name = str(model).split("(")[0]
        
        df_valid = data.set_index(self.date)

        if len(self.group_features) == 0:

            if seasonality:
                df_valid = self.create_seasonality(df_valid)

            if lag == False:

                dummy_vars = (self.categorical_features).copy()
                label_encode = []
                for col in dummy_vars:
                    if len(df_valid[col].unique()) > 500:
                        encoder = LabelEncoder()
                        df_valid[col] = encoder.fit_transform(df_valid[col])
                        label_encode = []

                for col in label_encode:
                    dummy_vars.pop(dummy_vars.index(col))

                df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                Xscaler = StandardScaler()
                yscaler = StandardScaler()

                df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                X_valid = df_valid[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])
                valid_index = X_valid.index
                if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                    X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                if 'Conv1D' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                    X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1)) 

                model_ = deepcopy(model)

                model_.fit(X_train, y_train, **fit_kwargs)

                try:
                    y_pred = model_.predict(X_valid, verbose = 0).flatten()
                except:
                    y_pred = model_.predict(X_valid).flatten()

                y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)

                y_pred = pd.Series(y_pred, index = valid_index)

                try:
                    y_train_pred = model_.predict(X_train, verbose = 0).flatten()
                except:
                    y_train_pred = model_.predict(X_train).flatten()

                training_score = self.scoring_metric(y_train, y_train_pred)

                scores = pd.DataFrame({model_name: [training_score]}, index = ['Training score']).T
                display(scores)

                if plot == True:
                    ax = y_train.plot(figsize = (15, 3))
                    y_pred.plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.keep_id:
                    y_pred_w_id = pd.DataFrame({self.id_: X_valid[id_], self.target: y_pred}, index = y_pred.index)
                    return y_pred_w_id, scores

                return y_pred.reset_index(), scores

            if lag != False:

                df_valid, lags = self.create_lags(df_valid, lag)

                dummy_vars = (self.categorical_features).copy()
                label_encode = []
                for col in dummy_vars:
                    if len(df_valid[col].unique()) > 500:
                        encoder = LabelEncoder()
                        df_valid[col] = encoder.fit_transform(df_valid[col])
                        label_encode = []

                for col in label_encode:
                    dummy_vars.pop(dummy_vars.index(col))

                df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                Xscaler = StandardScaler()
                yscaler = StandardScaler()

                df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

                if 'Conv1D' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))

                model_ = deepcopy(model)

                model_.fit(X_train, y_train, **fit_kwargs)

                try:
                    y_train_pred = model_.predict(X_train, verbose = 0).flatten()
                except:
                    y_train_pred = model_.predict(X_train).flatten()

                y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)

                training_score = self.scoring_metric(y_train, y_train_pred)

                forecast_index = df_valid[df_valid[self.target].isna()].index
                for i in tqdm_notebook(range(len(forecast_index))):

                    future_index = df_valid[df_valid[self.target].isna()].index[0]
                    future_X = df_valid.loc[future_index:future_index].drop(columns = [self.target, self.id_])
                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        future_X = np.reshape(future_X, (future_X.shape[0], 1, future_X.shape[1]))

                    if 'Conv1D' in str(model_name):
                        future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))
                    try:
                        future_1 = model_.predict(future_X, verbose = 0).flatten()[0]
                    except:
                        future_1 = model_.predict(future_X).flatten()[0]

                    df_valid.loc[future_index, self.target] = future_1

                    try:
                        unique_dates = pd.Series(df_valid.index.unique())
                        unique_index = np.where(unique_dates == future_index)[0][0]
                        for lag_ in lags:
                            df_valid.loc[unique_dates[unique_index + lag_], f'lag_{lag_}'] = future_1     
                    except:
                        pass

                y_pred_w_id = df_valid.loc[forecast_index, [self.id_, self.target]].copy()

                y_pred_w_id[self.target] = yscaler.inverse_transform(y_pred_w_id[self.target].values.reshape(-1, 1)).flatten()
               
                y_pred = y_pred_w_id[self.target]

                scores = pd.DataFrame({model_name: [training_score]}, index = ['Training score']).T
                display(scores)

                if plot == True:
                    ax = y_train.plot(figsize = (15, 3))
                    y_pred.plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.keep_id:
                    return y_pred_w_id, scores

                return y_pred.reset_index(), scores

        if by == None and len(self.group_features):

            if seasonality:
                df_valid = self.create_seasonality(df_valid)

            if lag == False:

                dummy_vars = (self.group_features + self.categorical_features).copy()
                label_encode = []
                for col in dummy_vars:
                    if len(df_valid[col].unique()) > 500:
                        encoder = LabelEncoder()
                        df_valid[col] = encoder.fit_transform(df_valid[col])
                        label_encode.append(col)

                for col in label_encode:
                    dummy_vars.pop(dummy_vars.index(col))          

                df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                Xscaler = StandardScaler()
                yscaler = StandardScaler()

                df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                X_valid = df_valid[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])
                valid_index = X_valid.index
                if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                    X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                if 'Conv1D' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                    X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1)) 

                model_ = deepcopy(model)

                model_.fit(X_train, y_train, **fit_kwargs)

                try:
                    y_pred = model_.predict(X_valid, verbose = 0).flatten()
                except:
                    y_pred = model_.predict(X_valid).flatten()

                y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)

                y_pred_w_id = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred}, index = valid_index)
                y_pred = y_pred_w_id.sort_values(by = [self.date, self.id_])[self.target]

                try:
                    y_train_pred = model_.predict(X_train, verbose = 0).flatten()
                except:
                    y_train_pred = model_.predict(X_train).flatten()
                
                training_score = self.scoring_metric(y_train, y_train_pred)

                scores = pd.DataFrame({model_name: [training_score]}, index = ['Training score']).T
                display(scores)

                if plot == True:
                    ax = y_train.reset_index().groupby(self.date).mean().plot(figsize = (15, 3))
                    y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.keep_id:
                    return y_pred_w_id, scores

                return y_pred.reset_index(), scores

            if lag != False:

                df_valid, lags = self.create_lags(df_valid, lag)

                dummy_vars = (self.group_features + self.categorical_features).copy()
                label_encode = []
                for col in dummy_vars:
                    if len(df_valid[col].unique()) > 500:
                        encoder = LabelEncoder()
                        df_valid[col] = encoder.fit_transform(df_valid[col])
                        label_encode = []

                for col in label_encode:
                    dummy_vars.pop(dummy_vars.index(col))

                df_valid_dummy = pd.get_dummies(df_valid, columns = dummy_vars)
                for feature in self.group_features:
                    df_valid_dummy[feature] = df_valid[feature]

                df_valid = df_valid_dummy
                del df_valid_dummy

                Xscaler = StandardScaler()
                yscaler = StandardScaler()

                df_valid[df_valid.columns.drop([self.id_, self.target, *self.group_features])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target, *self.group_features])])
                df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]
                if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

                if 'Conv1D' in str(model_name):
                    X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))

                model_ = deepcopy(model)

                model_.fit(X_train, y_train, **fit_kwargs)

                try:
                    y_train_pred = model_.predict(X_train, verbose = 0).flatten()
                except:
                    y_train_pred = model_.predict(X_train).flatten()
                
                y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
               
                training_score = self.scoring_metric(y_train, y_train_pred)

                y_pred_w_ids = []

                for group in tqdm_notebook(df_valid.groupby(self.group_features)):

                    df_group = group[1]

                    forecast_index = df_group[df_group[self.target].isna()].index
                    for i in range(len(forecast_index)):

                        future_index = df_group[df_group[self.target].isna()].index[0]
                        future_X = df_group.loc[future_index:future_index].drop(columns = [self.target, self.id_, *self.group_features])
                        if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                            future_X = np.reshape(future_X, (future_X.shape[0], 1, future_X.shape[1]))

                        if 'Conv1D' in str(model_name):
                            future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))
                        try:
                            future_1 = model_.predict(future_X, verbose = 0).flatten()[0]
                        except:
                            future_1 = model_.predict(future_X).flatten()[0]

                        df_group.loc[future_index, self.target] = future_1

                        try:
                            unique_dates = pd.Series(df_group.index.unique())
                            unique_index = np.where(unique_dates == future_index)[0][0]
                            for lag_ in lags:
                                df_group.loc[unique_dates[unique_index + lag_], f'lag_{lag_}'] = future_1                         
                        except:
                            pass

                    y_pred_w_ids.append(df_group.loc[forecast_index, [self.id_, self.target]].copy())

                y_pred_w_id = pd.concat(y_pred_w_ids).sort_values(by = [self.date, self.id_])

                y_pred_w_id[self.target] = yscaler.inverse_transform(y_pred_w_id[self.target].values.reshape(-1, 1)).flatten()

                y_pred = y_pred_w_id[self.target]

                scores = pd.DataFrame({model_name: [training_score]}, index = ['Training score']).T
                display(scores)

                if plot == True:
                    ax = y_train.reset_index().groupby(self.date).mean().plot(figsize = (15, 3))
                    y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.keep_id:
                    return y_pred_w_id, scores

                return y_pred.reset_index(), scores

        if len(by) == len(self.group_features):

            models = {}
            training_scores = {}
            test_scores = {}
            y_pred_w_ids = []
            y_preds = []
            y_trains = []

            for group in tqdm_notebook(data.groupby(self.group_features)):

                df_valid = group[1].set_index(self.date)

                if seasonality:
                    df_valid = self.create_seasonality(df_valid)

                if lag == False:

                    dummy_vars = (self.categorical_features).copy()
                    label_encode = []
                    for col in dummy_vars:
                        if len(df_valid[col].unique()) > 500:
                            encoder = LabelEncoder()
                            df_valid[col] = encoder.fit_transform(df_valid[col])
                            label_encode = []

                    for col in label_encode:
                        dummy_vars.pop(dummy_vars.index(col))                    

                    df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                    Xscaler = StandardScaler()
                    yscaler = StandardScaler()

                    df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                    df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]

                    X_valid = df_valid.drop(columns = self.group_features)[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])
                    valid_index = X_valid.index
                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                    if 'Conv1D' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1)) 

                    models[group[0]] = deepcopy(model)

                    models[group[0]].fit(X_train, y_train, **fit_kwargs)

                    try:
                        y_pred_ = models[group[0]].predict(X_valid, verbose = 0).flatten()
                    except:
                        y_pred_ = models[group[0]].predict(X_valid).flatten()

                    y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                    y_trains.append(y_train)

                    y_pred_w_id_ = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred_}, index = valid_index)
                    y_pred_w_ids.append(y_pred_w_id_)
                    y_pred_ = y_pred_w_id_.sort_values(by = [self.date, self.id_])[self.target]
                    y_preds.append(y_pred_)

                    try:
                        y_train_pred = models[group[0]].predict(X_train, verbose = 0).flatten()
                    except:
                        y_train_pred = models[group[0]].predict(X_train).flatten()

                    training_scores[group[0]] = (self.scoring_metric(y_train, y_train_pred))

                if lag != False:

                    df_valid, lags = self.create_lags(df_valid, lag)

                    dummy_vars = (self.categorical_features).copy()
                    label_encode = []
                    for col in dummy_vars:
                        if len(df_valid[col].unique()) > 500:
                            encoder = LabelEncoder()
                            df_valid[col] = encoder.fit_transform(df_valid[col])
                            label_encode = []

                    for col in label_encode:
                        dummy_vars.pop(dummy_vars.index(col))

                    df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                    Xscaler = StandardScaler()
                    yscaler = StandardScaler()

                    df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                    df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]
                    

                    X_valid = df_valid.drop(columns = self.group_features)[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])

                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                    if 'Conv1D' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1)) 

                    models[group[0]] = deepcopy(model)

                    models[group[0]].fit(X_train, y_train, **fit_kwargs)

                    try:
                        y_train_pred = models[group[0]].predict(X_train, verbose = 0).flatten()
                    except:
                        y_train_pred = models[group[0]].predict(X_train).flatten()

                    y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                    y_trains.append(y_train)
                                   
                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                    forecast_index = df_valid[df_valid[self.target].isna()].index
                    for i in (range(len(forecast_index))):

                        future_index = df_valid[df_valid[self.target].isna()].index[0]
                        future_X = df_valid.loc[future_index:future_index].drop(columns = [self.target, self.id_, *self.group_features])
                        if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                            future_X = np.reshape(future_X, (future_X.shape[0], 1, future_X.shape[1]))

                        if 'Conv1D' in str(model_name):
                            future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))
                        try:
                            future_1 = models[group[0]].predict(future_X, verbose = 0).flatten()[0]
                        except:
                            future_1 = models[group[0]].predict(future_X).flatten()[0]

                        df_valid.loc[future_index, self.target] = future_1

                        try:
                            unique_dates = pd.Series(df_valid.index.unique())
                            unique_index = np.where(unique_dates == future_index)[0][0]
                            for lag_ in lags:
                                df_valid.loc[unique_dates[unique_index + lag_], f'lag_{lag_}'] = future_1                           
                        except:
                            pass

                    y_pred_w_id_ = df_valid.loc[forecast_index, [self.id_, self.target]].copy()

                    y_pred_w_id[self.target] = yscaler.inverse_transform(y_pred_w_id[self.target].values.reshape(-1, 1)).flatten()

                    y_pred_ = y_pred_w_id_[self.target]

                    y_pred_w_ids.append(y_pred_w_id_)
                    y_preds.append(y_pred_)

            y_pred_w_id = pd.concat(y_pred_w_ids)
            y_pred = pd.concat(y_preds)

            scores = pd.DataFrame(training_scores, index = ['Training score']).T

            display(scores)
            display(scores.mean().to_frame().rename(columns = {0: 'AVERAGE'}).T)

            if plot == True:
                ax = pd.concat(y_trains).reset_index().groupby(self.date).mean().plot(figsize = (15, 3))
                y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                ax.set_xlabel(self.date)
                ax.set_ylabel(self.target)
                plt.tight_layout()
                plt.show()

            if self.keep_id:
                return y_pred_w_id, scores

            return y_pred.reset_index(), scores

        if len(by) == 1 and len(self.group_features) == 2:

            gr_1 = by
            gr_2 = self.group_features.copy()
            gr_2.pop(gr_2.index(by[0]))

            models = {}
            training_scores = {}
            test_scores = {}
            y_pred_w_ids = []
            y_preds = []
            y_trains = []

            for group in tqdm_notebook(data.groupby(gr_1)):

                df_valid = group[1].set_index(self.date)

                if seasonality:
                    df_valid = self.create_seasonality(df_valid)

                if lag == False:

                    dummy_vars = (gr_2 + self.categorical_features).copy()
                    label_encode = []
                    for col in dummy_vars:
                        if len(df_valid[col].unique()) > 500:
                            encoder = LabelEncoder()
                            df_valid[col] = encoder.fit_transform(df_valid[col])
                            label_encode = []

                    for col in label_encode:
                        dummy_vars.pop(dummy_vars.index(col))

                    df_valid = pd.get_dummies(df_valid, columns = dummy_vars)

                    Xscaler = StandardScaler()
                    yscaler = StandardScaler()

                    df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                    df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                    X_train = df_valid.drop(columns = by)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                    y_train = df_valid.drop(columns = by)[df_valid[self.target].notna()][self.target]

                    X_valid = df_valid.drop(columns = by)[df_valid[self.target].isna()].drop(columns = [self.target, self.id_])
                    valid_index = X_valid.index

                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], 1, X_valid.shape[1]))

                    if 'Conv1D' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))
                        X_valid = np.reshape(X_valid.values, (X_valid.shape[0], X_valid.shape[1], 1)) 

                    models[group[0]] = deepcopy(model)

                    models[group[0]].fit(X_train, y_train, **fit_kwargs)

                    try:
                        y_pred_ = models[group[0]].predict(X_valid, verbose = 0).flatten()
                    except:
                        y_pred_ = models[group[0]].predict(X_valid).flatten()

                    y_pred = yscaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                    y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                    y_trains.append(y_train)

                    y_pred_w_id_ = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred_}, index = valid_index)
                    y_pred_ = y_pred_w_id_.sort_values(by = [self.date, self.id_])[self.target]

                    y_pred_w_ids.append(y_pred_w_id_)
                    y_preds.append(y_pred_)

                    try:
                        y_train_pred = models[group[0]].predict(X_train, verbose = 0).flatten()
                    except:
                        y_train_pred = models[group[0]].predict(X_train).flatten()

                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                if lag != False:

                    df_valid, lags = self.create_lags(df_valid, lag)

                    dummy_vars = (gr_2 + self.categorical_features).copy()
                    label_encode = []
                    for col in dummy_vars:
                        if len(df_valid[col].unique()) > 500:
                            encoder = LabelEncoder()
                            df_valid[col] = encoder.fit_transform(df_valid[col])
                            label_encode = []

                    for col in label_encode:
                        dummy_vars.pop(dummy_vars.index(col))

                    df_valid_dummy = pd.get_dummies(df_valid, columns = dummy_vars)

                    for feature in gr_2:
                        df_valid_dummy[feature] = df_valid[feature]

                    df_valid = df_valid_dummy
                    del df_valid_dummy

                    Xscaler = StandardScaler()
                    yscaler = StandardScaler()

                    df_valid[df_valid.columns.drop([self.id_, self.target])] = Xscaler.fit_transform(df_valid[df_valid.columns.drop([self.id_, self.target])])
                    df_valid[self.target] = yscaler.fit_transform(df_valid[self.target].values.reshape(-1, 1)).flatten()

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = [self.target, self.id_])
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]

                    if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], 1, X_train.shape[1]))

                    if 'Conv1D' in str(model_name):
                        X_train = np.reshape(X_train.values, (X_train.shape[0], X_train.shape[1], 1))

                    models[group[0]] = deepcopy(model)

                    models[group[0]].fit(X_train, y_train, **fit_kwargs)

                    try:
                        y_train_pred = models[group[0]].predict(X_train, verbose = 0).flatten()
                    except:
                        y_train_pred = models[group[0]].predict(X_train).flatten()

                    y_train = pd.Series(yscaler.inverse_transform(y_train.values.reshape(-1, 1)).flatten(), index = y_train.index)
                    y_trains.append(y_train)
               
                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                    y_pred_w_ids_ = []

                    for group_ in df_valid.groupby(gr_2):

                        df_group = group_[1]

                        forecast_index = df_group[df_group[self.target].isna()].index
                        for i in range(len(forecast_index)):

                            future_index = df_group[df_group[self.target].isna()].index[0]
                            future_X = df_group.loc[future_index:future_index].drop(columns = [self.target, self.id_, *self.group_features])
                            if 'LSTM' in str(model_name) or 'GRU' in str(model_name):
                                future_X = np.reshape(future_X, (future_X.shape[0], 1, future_X.shape[1]))

                            if 'Conv1D' in str(model_name):
                                future_X = np.reshape(future_X, (future_X.shape[0], future_X.shape[1], 1))
                            try:
                                future_1 = models[group[0]].predict(future_X, verbose = 0).flatten()[0]
                            except:
                                future_1 = models[group[0]].predict(future_X).flatten()[0]

                            df_group.loc[future_index, self.target] = future_1

                            try:
                                unique_dates = pd.Series(df_group.index.unique())
                                unique_index = np.where(unique_dates == future_index)[0][0]
                                for lag_ in lags:
                                    df_group.loc[unique_dates[unique_index + lag_], f'lag_{lag_}'] = future_1                             
                            except:
                                pass

                        y_pred_w_id__ = df_group.loc[forecast_index, [self.id_, self.target]].copy()
                        y_pred_w_ids_.append(y_pred_w_id__)

                    y_pred_w_id_ = pd.concat(y_pred_w_ids_).sort_values(by = [self.date, self.id_])

                    y_pred_w_id_[self.target] = yscaler.inverse_transform(y_pred_w_id_[self.target].values.reshape(-1, 1)).flatten()

                    y_pred_ = y_pred_w_id_[self.target]

                    y_pred_w_ids.append(y_pred_w_id_)
                    y_preds.append(y_pred_)

            y_pred_w_id = pd.concat(y_pred_w_ids)
            y_pred = pd.concat(y_preds)

            scores = pd.DataFrame(training_scores, index = ['Training score']).T

            display(scores)
            display(scores.mean().to_frame().rename(columns = {0: 'AVERAGE'}).T)

            if plot == True:
                ax = pd.concat(y_trains).reset_index().groupby(self.date).mean().plot(figsize = (15, 3))
                y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                ax.set_xlabel(self.date)
                ax.set_ylabel(self.target)
                plt.tight_layout()
                plt.show()

            if self.keep_id:
                return y_pred_w_id, scores

            return y_pred.reset_index(), scores
