import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from kaggle.competitions import twosigmanews


env = twosigmanews.make_env()


def add_date(market_df, news_df):
    for df in [market_df, news_df]:
        df['date'] = pd.to_datetime(df['time'].dt.date)
    return market_df, news_df


def get_training_data():
    market_train_df, news_train_df = add_date(env.get_training_data())
    return market_train_df, news_train_df


def validate(model, market_df, news_df, n_splits=3, return_daily_validation_info=False):
    dates = np.array(sorted(market_df['date'].unique()))
    values = []
    daily_info = []
    for train_index, test_index in TimeSeriesSplit(n_splits=n_splits).split(dates):
        train_dates = dates[train_index]
        train_before = train_dates.max()

        market_train_df = market_df.loc[market_df['date'] <= train_before]
        news_train_df = news_df.loc[news_df['date'] <= train_before]
        market_val_df = market_df.loc[market_df['date'] > train_before]
        news_val_df = news_df.loc[news_df['date'] > train_before]

        model.fit(market_train_df, news_train_df)
        prediction = model.predict(market_val_df, news_val_df)
        validation_info = market_val_df[['assetCode', 'date', 'returnsOpenNextMktres10', 'universe']].merge(
            prediction,
            left_on=['assetCode', 'date'],
            right_on=['assetCode', 'date'],
        )
        validation_info['metric'] = validation_info['confidenceValue'] * \
            validation_info['returnsOpenNextMktres10'] * \
            validation_info['universe']
        validation_info_grouped = validation_info.groupby('date')[['metric']].sum()
        if validation_info_grouped['metric'].std() == 0.0:
            metric = 0.0
        else:
            metric = validation_info_grouped['metric'].mean() / validation_info_grouped['metric'].std()
        values.append(metric)
        if return_daily_validation_info:
            daily_info.append(validation_info)
    metric_values = np.array(values)
    if return_daily_validation_info:
        daily_info = pd.concat(daily_info, sort=True).sort_values('date')
        return metric_values, daily_info
    else:
        return metric_values


def make_submission(model, market_train_df, news_train_df):
    model.fit(market_train_df, news_train_df)
    for market_obs_df, news_obs_df, predictions_template_df in env.get_prediction_days():
        market_obs_df['date'] = pd.to_datetime(market_obs_df['time'].dt.date)
        news_obs_df['date'] = pd.to_datetime(news_obs_df['time'].dt.date)
        predictions_df = model.predict(market_obs_df, news_obs_df)[['assetCode', 'confidenceValue']]
        env.predict(predictions_df)
    env.write_submission_file()
