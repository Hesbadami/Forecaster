import warnings
warnings.simplefilter(action='ignore')

import pandas as pd
pd.set_option('display.max_columns', None)

import numpy as np
import matplotlib.pyplot as plt
plt.style.use('bmh')

from scipy.signal import periodogram
from statsmodels.tsa.stattools import pacf, acf
from sklearn import metrics
from tqdm import tqdm_notebook

class Forecaster:

    def __init__(self, df, date_and_target, group_features = [], categorical_features = [], scoring_metric = metrics.mean_squared_error, id_ = None):
        self.df = df

        self.id_ = id_
        if not self.id_:
            self.id_ = 'id'
            self.df['id'] = 0

        self.date, self.target = date_and_target
        self.group_features = group_features
        self.categorical_features = categorical_features
        self.scoring_metric = scoring_metric
        self.time_delta = pd.Timedelta(self.df[self.date].unique()[1] - self.df[self.date].unique()[0])

    def make_future_dataframe(self, periods, fill_zero = []):

        future = pd.DataFrame()
        
        start_date = self.df[self.date].iloc[-1] + self.time_delta
        end_date = start_date + (periods * self.time_delta)
        date_range = pd.date_range(start_date)
        
        to_copy = self.df[self.df[self.date] == self.df[self.date].iloc[-1]].copy()
        
        for date in date_range:
            to_copy[self.date] = date
            to_copy['date_index'] += 1
            
            # to_copy[to_copy.columns.drop([self.id_, self.date, 'date_index', *self.group_features, *self.categorical_features])] = 0
            to_copy[fill_zero] = 0
            
            for col in to_copy.columns:
                if col.lower() == 'year':
                    to_copy[col] = to_copy[self.date].dt.year
                if col.lower() == 'month':
                    to_copy[col] = to_copy[self.date].dt.month
                if col.lower() in ['day_of_week', 'dayofweek']:
                    to_copy[col] = to_copy[self.date].dt.dayofweek
                if col.lower() in ['day_of_month', 'dayofmonth']:
                    to_copy[col] = to_copy[self.date].dt.day
                if col.lower() in ['day_of_year', 'dayofyear']:
                    to_copy[col] = to_copy[self.date].dt.dayofyear
                if col.lower() in ['week_of_year', 'weekofyear', 'week']:
                    to_copy[col] = to_copy[self.date].dt.isocalendar().week.astype(int)
            
            future = pd.concat([future, to_copy])
            
        self.df = pd.concat([future, self.df])
    
    def create_seasonality(self, data):

        ret = pd.DataFrame()

        fs = pd.Timedelta(365, 'd')/self.time_delta

        freqencies, spectrum = periodogram(
            data[data[self.target].notna()].groupby(self.date)[self.target].mean(),
            fs=fs,
            detrend='linear',
            window="boxcar",
            scaling='spectrum',
        )

        Seasonality = np.round(fs/freqencies[spectrum > spectrum.max()/5], 2)

        for s in Seasonality:
            ret[f'sin_{s}'] = np.sin(data['date_index'] * (2*np.pi / s))
            ret[f'cos_{s}'] = np.cos(data['date_index'] * (2*np.pi / s))

        return pd.concat([data, ret], axis = 1)

    def create_lags(self, data, lags = True):

        ret = pd.DataFrame()

        if lags == True:
            average_sales = data[data[self.target].notna()].groupby(self.date)[self.target].mean()
            plt.show()
            alpha=0.05
            method="ywm"
            nlags = len(average_sales)//2-1
            lags, thresh = acf(average_sales, nlags=nlags, alpha=alpha)#, method=method)
            lags = np.where(((lags > thresh[:, 1] - lags)&(lags > 0.3)) | ((lags < thresh[:, 0] - lags)&(lags < -0.3)))[0][1:]

        if len(lags):

            if len(self.group_features):
                for lag in lags:
                    ret[f'lag_{lag}'] = data.groupby(self.group_features)[self.target].transform(lambda x: x.shift(lag))

            elif len(self.group_features) == 0:
                for lag in lags:
                    ret[f'lag_{lag}'] = data[self.target].shift(lag)

            ret = pd.concat([data, ret], axis = 1)
            nan_head = ret.index[0] + (lags[-1] * self.time_delta)

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

    def validate(self, model, seasonality = False, lag = False, by = None, plot = False):

        model_name = str(model).split(".")[-1].split("'")[0]

        if len(self.group_features) == 0:
            df_valid , y_valid = self.train_valid_split(self.df[self.df[self.target].notna()].set_index(self.date))

            if seasonality:
                df_valid = self.create_seasonality(df_valid)

            if lag == False:

                df_valid = pd.get_dummies(df_valid, columns = self.categorical_features)

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = self.target)
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                X_valid = df_valid[df_valid[self.target].isna()].drop(columns = self.target)

                model_ = model()
                model_.fit(X_train.drop(columns = self.id_), y_train)

                y_pred = model_.predict(X_valid.drop(columns = self.id_))
                y_pred[y_pred < 0] = 0
                y_pred = pd.Series(y_pred, index = X_valid.index)

                y_train_pred = model_.predict(X_train.drop(columns = 'id'))
                y_train_pred[y_train_pred < 0] = 0
                training_score = self.scoring_metric(y_train, y_train_pred)
                test_score = self.scoring_metric(y_valid, y_pred)

                scores = pd.DataFrame({model_name: [training_score, test_score]}, index = ['Training score', 'Test score']).T
                display(scores)

                if plot == True:
                    ax = y_train.plot(figsize = (15, 3))
                    y_valid.plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                    y_pred.plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.id_:
                    y_pred_w_id = pd.DataFrame({self.id_: X_valid[id_], self.target: y_pred}, index = y_pred.index)
                    return y_pred_w_id, scores

                return y_pred, scores

            if lag != False:

                df_valid, lags = self.create_lags(df_valid, lag)
                df_valid = pd.get_dummies(df_valid, columns = self.categorical_features)

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = self.target)
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                model_ = model()
                model_.fit(X_train.drop(columns = self.id_), y_train)

                y_train_pred = model_.predict(X_train.drop(columns = self.id_))
                y_train_pred[y_train_pred < 0] = 0
                training_score = self.scoring_metric(y_train, y_train_pred)

                forecast_index = df_valid[df_valid[self.target].isna()].index
                for i in tqdm_notebook(range(len(forecast_index))):

                    future_index = df_valid[df_valid[self.target].isna()].index[0]
                    future_X = df_valid.loc[future_index:future_index].drop(columns = self.target)

                    future_1 = model_.predict(future_X.drop(columns = self.id_))[0]

                    df_valid.loc[future_index, self.target] = future_1
                    for lag_ in lags:
                        df_valid.loc[future_index + (lag_ * self.time_delta), f'lag_{lag_}'] = future_1

                y_pred_w_id = df_valid.loc[forecast_index, [self.id_, self.target]].copy()
                y_pred_w_id.loc[y_pred_w_id[self.target] < 0, self.target] = 0

                y_pred = y_pred_w_id[self.target]

                test_score = self.scoring_metric(y_valid, y_pred)

                scores = pd.DataFrame({model_name: [training_score, test_score]}, index = ['Training score', 'Test score']).T
                display(scores)

                if plot == True:
                    ax = y_train.plot(figsize = (15, 3))
                    y_valid.plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                    y_pred.plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.id_:
                    return y_pred_w_id, scores

                return y_pred, scores

        if by == None and len(self.group_features):
            df_valid , y_valid = self.train_valid_split(self.df[self.df[self.target].notna()].set_index(self.date))

            if seasonality:
                df_valid = self.create_seasonality(df_valid)

            if lag == False:

                df_valid = pd.get_dummies(df_valid, columns = self.group_features + self.categorical_features)

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = self.target)
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                X_valid = df_valid[df_valid[self.target].isna()].drop(columns = self.target)

                model_ = model()
                model_.fit(X_train.drop(columns = self.id_), y_train)

                y_pred = model_.predict(X_valid.drop(columns = self.id_))
                y_pred[y_pred < 0] = 0

                y_pred_w_id = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred}, index = X_valid.index)
                y_pred = y_pred_w_id.sort_values(by = [self.date, self.id_])[self.target]

                y_train_pred = model_.predict(X_train.drop(columns = self.id_))
                y_train_pred[y_train_pred < 0] = 0
                training_score = self.scoring_metric(y_train, y_train_pred)
                test_score = self.scoring_metric(y_valid, y_pred)

                scores = pd.DataFrame({model_name: [training_score, test_score]}, index = ['Training score', 'Test score']).T
                display(scores)

                if plot == True:
                    ax = y_train.reset_index().groupby(self.date).mean().plot(figsize = (15, 3))
                    y_valid.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                    y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.id_:
                    return y_pred_w_id, scores

                return y_pred, scores

            if lag != False:

                df_valid, lags = self.create_lags(df_valid, lag)
                df_valid_dummy = pd.get_dummies(df_valid, columns = self.group_features + self.categorical_features)
                for feature in self.group_features:
                    df_valid_dummy[feature] = df_valid[feature]

                df_valid = df_valid_dummy
                del df_valid_dummy

                X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = self.target)
                y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]

                model_ = model()
                model_.fit(X_train.drop(columns = self.id_), y_train)

                y_train_pred = model_.predict(X_train.drop(columns = self.id_))
                y_train_pred[y_train_pred < 0] = 0
                training_score = self.scoring_metric(y_train, y_train_pred)

                y_pred_w_ids = []

                for group in tqdm_notebook(df_valid.groupby(self.group_features)):

                    df_group = group[1]

                    forecast_index = df_group[df_group[self.target].isna()].index
                    for i in range(len(forecast_index)):

                        future_index = df_group[df_group[self.target].isna()].index[0]
                        future_X = df_group.loc[future_index:future_index].drop(columns = self.target)

                        future_1 = model_.predict(future_X.drop(columns = [self.id_, *self.group_features]))[0]

                        df_group.loc[future_index, self.target] = future_1
                        for lag_ in lags:
                            df_group.loc[future_index + (lag_ * self.time_delta), f'lag_{lag_}'] = future_1

                    y_pred_w_ids.append(df_group.loc[forecast_index, [self.id_, self.target]].copy())

                y_pred_w_id = pd.concat(y_pred_w_ids).sort_values(by = [self.date, self.id_])
                y_pred_w_id.loc[y_pred_w_id[self.target] < 0, self.target] = 0

                y_pred = y_pred_w_id[self.target]

                test_score = self.scoring_metric(y_valid, y_pred)

                scores = pd.DataFrame({model_name: [training_score, test_score]}, index = ['Training score', 'Test score']).T
                display(scores)

                if plot == True:
                    ax = y_train.reset_index().groupby(self.date).mean().plot(figsize = (15, 3))
                    y_valid.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:blue', alpha = 0.5)
                    y_pred.reset_index().groupby(self.date).mean().plot(ax = ax, color = 'tab:orange')
                    ax.set_xlabel(self.date)
                    ax.set_ylabel(self.target)
                    plt.tight_layout()
                    plt.show()

                if self.id_:
                    return y_pred_w_id, scores

                return y_pred, scores

        if len(by) == len(self.group_features):

            models = {}
            training_scores = {}
            test_scores = {}
            y_pred_w_ids = []
            y_preds = []
            y_trains = []
            y_valids = []

            for group in tqdm_notebook(self.df.groupby(self.group_features)):

                df_valid , y_valid = self.train_valid_split(group[1][group[1][self.target].notna()].set_index(self.date))
                y_valids.append(y_valid)

                if seasonality:
                    df_valid = self.create_seasonality(df_valid)

                if lag == False:

                    df_valid = pd.get_dummies(df_valid, columns = self.categorical_features)

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = self.target)
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]
                    y_trains.append(y_train)

                    X_valid = df_valid.drop(columns = self.group_features)[df_valid[self.target].isna()].drop(columns = self.target)

                    models[group[0]] = model()
                    models[group[0]].fit(X_train.drop(columns = self.id_), y_train)

                    y_pred_ = models[group[0]].predict(X_valid.drop(columns = self.id_))
                    y_pred_[y_pred_ < 0] = 0

                    y_pred_w_id_ = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred_}, index = X_valid.index)
                    y_pred_w_ids.append(y_pred_w_id_)
                    y_pred_ = y_pred_w_id_.sort_values(by = [self.date, self.id_])[self.target]
                    y_preds.append(y_pred_)

                    y_train_pred = models[group[0]].predict(X_train.drop(columns = self.id_))
                    y_train_pred[y_train_pred < 0] = 0
                    training_scores[group[0]] = (self.scoring_metric(y_train, y_train_pred))
                    test_scores[group[0]] = (self.scoring_metric(y_valid, y_pred_))

                if lag != False:

                    df_valid, lags = self.create_lags(df_valid, lag)
                    df_valid = pd.get_dummies(df_valid, columns = self.categorical_features)

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = self.target)
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]
                    y_trains.append(y_train)

                    X_valid = df_valid.drop(columns = self.group_features)[df_valid[self.target].isna()].drop(columns = self.target)

                    models[group[0]] = model()
                    models[group[0]].fit(X_train.drop(columns = self.id_), y_train)

                    y_train_pred = models[group[0]].predict(X_train.drop(columns = self.id_))
                    y_train_pred[y_train_pred < 0] = 0
                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                    forecast_index = df_valid[df_valid[self.target].isna()].index
                    for i in (range(len(forecast_index))):

                        future_index = df_valid[df_valid[self.target].isna()].index[0]
                        future_X = df_valid.loc[future_index:future_index].drop(columns = self.target)

                        future_1 = models[group[0]].predict(future_X.drop(columns = [self.id_, *self.group_features]))[0]

                        df_valid.loc[future_index, self.target] = future_1
                        for lag_ in lags:
                            df_valid.loc[future_index + (lag_ * self.time_delta), f'lag_{lag_}'] = future_1

                    y_pred_w_id_ = df_valid.loc[forecast_index, [self.id_, self.target]].copy()
                    y_pred_w_id_.loc[y_pred_w_id_[self.target] < 0, self.target] = 0

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

            if self.id_:
                return y_pred_w_id, scores

            return y_pred, scores

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
            y_valids = []

            for group in tqdm_notebook(self.df.groupby(gr_1)):

                df_valid , y_valid = self.train_valid_split(group[1][group[1][self.target].notna()].set_index(self.date))
                y_valids.append(y_valid)

                if seasonality:
                    df_valid = self.create_seasonality(df_valid)

                if lag == False:

                    df_valid = pd.get_dummies(df_valid, columns = gr_2 + self.categorical_features)

                    X_train = df_valid.drop(columns = by)[df_valid[self.target].notna()].drop(columns = self.target)
                    y_train = df_valid.drop(columns = by)[df_valid[self.target].notna()][self.target]
                    y_trains.append(y_train)

                    X_valid = df_valid.drop(columns = by)[df_valid[self.target].isna()].drop(columns = self.target)

                    models[group[0]] = model()
                    models[group[0]].fit(X_train.drop(columns = self.id_), y_train)

                    y_pred_ = models[group[0]].predict(X_valid.drop(columns = self.id_))
                    y_pred_[y_pred_ < 0] = 0

                    y_pred_w_id_ = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred_}, index = X_valid.index)
                    y_pred_ = y_pred_w_id_.sort_values(by = [self.date, self.id_])[self.target]

                    y_pred_w_ids.append(y_pred_w_id_)
                    y_preds.append(y_pred_)

                    y_train_pred = models[group[0]].predict(X_train.drop(columns = self.id_))
                    y_train_pred[y_train_pred < 0] = 0
                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)
                    test_scores[group[0]] = self.scoring_metric(y_valid, y_pred_)

                if lag != False:

                    df_valid, lags = self.create_lags(df_valid, lag)
                    df_valid_dummy = pd.get_dummies(df_valid, columns = gr_2 + self.categorical_features)

                    for feature in gr_2:
                        df_valid_dummy[feature] = df_valid[feature]

                    df_valid = df_valid_dummy
                    del df_valid_dummy

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = self.target)
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]
                    y_trains.append(y_train)

                    models[group[0]] = model()
                    models[group[0]].fit(X_train.drop(columns = self.id_), y_train)

                    y_train_pred = models[group[0]].predict(X_train.drop(columns = self.id_))
                    y_train_pred[y_train_pred < 0] = 0
                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                    y_pred_w_ids_ = []

                    for group_ in df_valid.groupby(gr_2):

                        df_group = group_[1]

                        forecast_index = df_group[df_group[self.target].isna()].index
                        for i in range(len(forecast_index)):

                            future_index = df_group[df_group[self.target].isna()].index[0]
                            future_X = df_group.loc[future_index:future_index].drop(columns = self.target)

                            future_1 = models[group[0]].predict(future_X.drop(columns = [self.id_, *self.group_features]))[0]

                            df_group.loc[future_index, self.target] = future_1
                            for lag_ in lags:
                                df_group.loc[future_index + (lag_ * self.time_delta), f'lag_{lag_}'] = future_1

                        y_pred_w_id__ = df_group.loc[forecast_index, [self.id_, self.target]].copy()
                        y_pred_w_ids_.append(y_pred_w_id__)

                    y_pred_w_id_ = pd.concat(y_pred_w_ids_).sort_values(by = [self.date, self.id_])
                    y_pred_w_id_.loc[y_pred_w_id_[self.target] < 0, self.target] = 0

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

            if self.id_:
                return y_pred_w_id, scores

            return y_pred, scores


    def forecast(self, model, seasonality = False, lag = False, by = None, plot = False):

        model_name = str(model).split(".")[-1].split("'")[0]
        df_valid = self.df.copy().set_index(self.date)

        if len(self.group_features) == 0:

            if seasonality:
                df_valid = self.create_seasonality(df_valid)

            if lag == False:

                df_valid = pd.get_dummies(df_valid, columns = self.categorical_features)

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = self.target)
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                X_valid = df_valid[df_valid[self.target].isna()].drop(columns = self.target)

                model_ = model()
                model_.fit(X_train.drop(columns = self.id_), y_train)

                y_pred = model_.predict(X_valid.drop(columns = self.id_))
                y_pred[y_pred < 0] = 0
                y_pred = pd.Series(y_pred, index = X_valid.index)

                y_train_pred = model_.predict(X_train.drop(columns = 'id'))
                y_train_pred[y_train_pred < 0] = 0
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

                if self.id_:
                    y_pred_w_id = pd.DataFrame({self.id_: X_valid[id_], self.target: y_pred}, index = y_pred.index)
                    return y_pred_w_id, scores

                return y_pred, scores

            if lag != False:

                df_valid, lags = self.create_lags(df_valid, lag)
                df_valid = pd.get_dummies(df_valid, columns = self.categorical_features)

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = self.target)
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                model_ = model()
                model_.fit(X_train.drop(columns = self.id_), y_train)

                y_train_pred = model_.predict(X_train.drop(columns = self.id_))
                y_train_pred[y_train_pred < 0] = 0
                training_score = self.scoring_metric(y_train, y_train_pred)

                forecast_index = df_valid[df_valid[self.target].isna()].index
                for i in tqdm_notebook(range(len(forecast_index))):

                    future_index = df_valid[df_valid[self.target].isna()].index[0]
                    future_X = df_valid.loc[future_index:future_index].drop(columns = self.target)

                    future_1 = model_.predict(future_X.drop(columns = self.id_))[0]

                    df_valid.loc[future_index, self.target] = future_1
                    for lag_ in lags:
                        df_valid.loc[future_index + (lag_ * self.time_delta), f'lag_{lag_}'] = future_1

                y_pred_w_id = df_valid.loc[forecast_index, [self.id_, self.target]].copy()
                y_pred_w_id.loc[y_pred_w_id[self.target] < 0, self.target] = 0

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

                if self.id_:
                    return y_pred_w_id, scores

                return y_pred, scores

        if by == None and len(self.group_features):

            if seasonality:
                df_valid = self.create_seasonality(df_valid)

            if lag == False:

                df_valid = pd.get_dummies(df_valid, columns = self.group_features + self.categorical_features)

                X_train = df_valid[df_valid[self.target].notna()].drop(columns = self.target)
                y_train = df_valid[df_valid[self.target].notna()][self.target]

                X_valid = df_valid[df_valid[self.target].isna()].drop(columns = self.target)

                model_ = model()
                model_.fit(X_train.drop(columns = self.id_), y_train)

                y_pred = model_.predict(X_valid.drop(columns = self.id_))
                y_pred[y_pred < 0] = 0

                y_pred_w_id = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred}, index = X_valid.index)
                y_pred = y_pred_w_id.sort_values(by = [self.date, self.id_])[self.target]

                y_train_pred = model_.predict(X_train.drop(columns = self.id_))
                y_train_pred[y_train_pred < 0] = 0
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

                if self.id_:
                    return y_pred_w_id, scores

                return y_pred, scores

            if lag != False:

                df_valid, lags = self.create_lags(df_valid, lag)
                df_valid_dummy = pd.get_dummies(df_valid, columns = self.group_features + self.categorical_features)
                for feature in self.group_features:
                    df_valid_dummy[feature] = df_valid[feature]

                df_valid = df_valid_dummy
                del df_valid_dummy

                X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = self.target)
                y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]

                model_ = model()
                model_.fit(X_train.drop(columns = self.id_), y_train)

                y_train_pred = model_.predict(X_train.drop(columns = self.id_))
                y_train_pred[y_train_pred < 0] = 0
                training_score = self.scoring_metric(y_train, y_train_pred)

                y_pred_w_ids = []

                for group in tqdm_notebook(df_valid.groupby(self.group_features)):

                    df_group = group[1]

                    forecast_index = df_group[df_group[self.target].isna()].index
                    for i in range(len(forecast_index)):

                        future_index = df_group[df_group[self.target].isna()].index[0]
                        future_X = df_group.loc[future_index:future_index].drop(columns = self.target)

                        future_1 = model_.predict(future_X.drop(columns = [self.id_, *self.group_features]))[0]

                        df_group.loc[future_index, self.target] = future_1
                        for lag_ in lags:
                            df_group.loc[future_index + (lag_ * self.time_delta), f'lag_{lag_}'] = future_1

                    y_pred_w_ids.append(df_group.loc[forecast_index, [self.id_, self.target]].copy())

                y_pred_w_id = pd.concat(y_pred_w_ids).sort_values(by = [self.date, self.id_])
                y_pred_w_id.loc[y_pred_w_id[self.target] < 0, self.target] = 0

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

                if self.id_:
                    return y_pred_w_id, scores

                return y_pred, scores

        if len(by) == len(self.group_features):

            models = {}
            training_scores = {}
            test_scores = {}
            y_pred_w_ids = []
            y_preds = []
            y_trains = []

            for group in tqdm_notebook(self.df.groupby(self.group_features)):

                df_valid = group[1].set_index(self.date)

                if seasonality:
                    df_valid = self.create_seasonality(df_valid)

                if lag == False:

                    df_valid = pd.get_dummies(df_valid, columns = self.categorical_features)

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = self.target)
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]
                    y_trains.append(y_train)

                    X_valid = df_valid.drop(columns = self.group_features)[df_valid[self.target].isna()].drop(columns = self.target)

                    models[group[0]] = model()
                    models[group[0]].fit(X_train.drop(columns = self.id_), y_train)

                    y_pred_ = models[group[0]].predict(X_valid.drop(columns = self.id_))
                    y_pred_[y_pred_ < 0] = 0

                    y_pred_w_id_ = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred_}, index = X_valid.index)
                    y_pred_w_ids.append(y_pred_w_id_)
                    y_pred_ = y_pred_w_id_.sort_values(by = [self.date, self.id_])[self.target]
                    y_preds.append(y_pred_)

                    y_train_pred = models[group[0]].predict(X_train.drop(columns = self.id_))
                    y_train_pred[y_train_pred < 0] = 0
                    training_scores[group[0]] = (self.scoring_metric(y_train, y_train_pred))

                if lag != False:

                    df_valid, lags = self.create_lags(df_valid, lag)
                    df_valid = pd.get_dummies(df_valid, columns = self.categorical_features)

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = self.target)
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]
                    y_trains.append(y_train)

                    X_valid = df_valid.drop(columns = self.group_features)[df_valid[self.target].isna()].drop(columns = self.target)

                    models[group[0]] = model()
                    models[group[0]].fit(X_train.drop(columns = self.id_), y_train)

                    y_train_pred = models[group[0]].predict(X_train.drop(columns = self.id_))
                    y_train_pred[y_train_pred < 0] = 0
                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                    forecast_index = df_valid[df_valid[self.target].isna()].index
                    for i in (range(len(forecast_index))):

                        future_index = df_valid[df_valid[self.target].isna()].index[0]
                        future_X = df_valid.loc[future_index:future_index].drop(columns = self.target)

                        future_1 = models[group[0]].predict(future_X.drop(columns = [self.id_, *self.group_features]))[0]

                        df_valid.loc[future_index, self.target] = future_1
                        for lag_ in lags:
                            df_valid.loc[future_index + (lag_ * self.time_delta), f'lag_{lag_}'] = future_1

                    y_pred_w_id_ = df_valid.loc[forecast_index, [self.id_, self.target]].copy()
                    y_pred_w_id_.loc[y_pred_w_id_[self.target] < 0, self.target] = 0

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

            if self.id_:
                return y_pred_w_id, scores

            return y_pred, scores

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

            for group in tqdm_notebook(self.df.groupby(gr_1)):

                df_valid = group[1].set_index(self.date)

                if seasonality:
                    df_valid = self.create_seasonality(df_valid)

                if lag == False:

                    df_valid = pd.get_dummies(df_valid, columns = gr_2 + self.categorical_features)

                    X_train = df_valid.drop(columns = by)[df_valid[self.target].notna()].drop(columns = self.target)
                    y_train = df_valid.drop(columns = by)[df_valid[self.target].notna()][self.target]
                    y_trains.append(y_train)

                    X_valid = df_valid.drop(columns = by)[df_valid[self.target].isna()].drop(columns = self.target)

                    models[group[0]] = model()
                    models[group[0]].fit(X_train.drop(columns = self.id_), y_train)

                    y_pred_ = models[group[0]].predict(X_valid.drop(columns = self.id_))
                    y_pred_[y_pred_ < 0] = 0

                    y_pred_w_id_ = pd.DataFrame({self.id_: X_valid[self.id_], self.target: y_pred_}, index = X_valid.index)
                    y_pred_ = y_pred_w_id_.sort_values(by = [self.date, self.id_])[self.target]

                    y_pred_w_ids.append(y_pred_w_id_)
                    y_preds.append(y_pred_)

                    y_train_pred = models[group[0]].predict(X_train.drop(columns = self.id_))
                    y_train_pred[y_train_pred < 0] = 0
                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                if lag != False:

                    df_valid, lags = self.create_lags(df_valid, lag)
                    df_valid_dummy = pd.get_dummies(df_valid, columns = gr_2 + self.categorical_features)

                    for feature in gr_2:
                        df_valid_dummy[feature] = df_valid[feature]

                    df_valid = df_valid_dummy
                    del df_valid_dummy

                    X_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()].drop(columns = self.target)
                    y_train = df_valid.drop(columns = self.group_features)[df_valid[self.target].notna()][self.target]
                    y_trains.append(y_train)

                    models[group[0]] = model()
                    models[group[0]].fit(X_train.drop(columns = self.id_), y_train)

                    y_train_pred = models[group[0]].predict(X_train.drop(columns = self.id_))
                    y_train_pred[y_train_pred < 0] = 0
                    training_scores[group[0]] = self.scoring_metric(y_train, y_train_pred)

                    y_pred_w_ids_ = []

                    for group_ in df_valid.groupby(gr_2):

                        df_group = group_[1]

                        forecast_index = df_group[df_group[self.target].isna()].index
                        for i in range(len(forecast_index)):

                            future_index = df_group[df_group[self.target].isna()].index[0]
                            future_X = df_group.loc[future_index:future_index].drop(columns = self.target)

                            future_1 = models[group[0]].predict(future_X.drop(columns = [self.id_, *self.group_features]))[0]

                            df_group.loc[future_index, self.target] = future_1
                            for lag_ in lags:
                                df_group.loc[future_index + (lag_ * self.time_delta), f'lag_{lag_}'] = future_1

                        y_pred_w_id__ = df_group.loc[forecast_index, [self.id_, self.target]].copy()
                        y_pred_w_ids_.append(y_pred_w_id__)

                    y_pred_w_id_ = pd.concat(y_pred_w_ids_).sort_values(by = [self.date, self.id_])
                    y_pred_w_id_.loc[y_pred_w_id_[self.target] < 0, self.target] = 0

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

            if self.id_:
                return y_pred_w_id, scores

            return y_pred, scores
