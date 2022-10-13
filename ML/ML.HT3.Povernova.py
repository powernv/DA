import streamlit as st
import os
import pandas as pd
import datetime
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

ts = st.slider('Выберите размер test_size', 0, 0.3)

date = st.date_input('Выберите день для прогноза курса EUR/RUB:', 
                     value = [datetime.date(2022, 10, 5), datetime.date(2022, 10, 6)], 
                     min_value = datetime.date(2022, 10, 5), 
                     max_value = datetime.date(2022, 10, 6))
dd = max(date)
st.write('Прогноз курса отразится по состоянию на:', dd)

if st.button('Показать данные курса валют EUR/RUB'):
    df = pd.read_csv(os.path.join('EUR_RUB__TOM.csv'), sep = ';')
    df.drop(['<TICKER>', '<PER>', '<TIME>', '<VOL>'], axis = 1, inplace = True)
    df = df.rename(columns = {'<DATE>':'dDate', '<CLOSE>':'nPr'})
    df.dDate = pd.to_datetime(df['dDate'], format='%Y%m%d')
    df = df[:(len(df)-1)]
    df.set_index(['dDate'], inplace = True)
    st.dataframe(df)
    
if st.button('Построить график по первоначальным данным курса валют EUR/RUB'):
    st.line_chart(df, x = 'Дата', y = 'Курс валют EUR/RUB')
      
if st.button('Обучить модель'):
    df = pd.read_csv(os.path.join('EUR_RUB__TOM.csv'), sep = ';')
    df.drop(['<TICKER>', '<PER>', '<TIME>', '<VOL>'], axis = 1, inplace = True)
    df = df.rename(columns = {'<DATE>':'dDate', '<CLOSE>':'nPr'})
    df.dDate = pd.to_datetime(df['dDate'], format='%Y%m%d')
    df = df[:(len(df)-1)]
    df.set_index(['dDate'], inplace = True)
    
    def add_features(df, max_lag, rolling_mean_size):
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_week'] = df.index.dayofweek
        
        for lag in range(1, max_lag + 1):
            df[f'lag_{lag}'] = df['nPr'].shift(lag)
            
        df['y_mean'] = df['nPr'].shift().rolling(rolling_mean_size).mean().copy()
        
        
    add_features(df, 40, 3) 
    df.dropna(inplace = True)
    
    
    X_train,X_test, y_train, y_test = train_test_split(df.drop('nPr', axis = 1), 
                                                       df.nPr, 
                                                       shuffle = False, 
                                                       test_size = ts, 
                                                       random_state = 35)
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr.predict(X_test)
    st.text('Качество модели ' + str(round(mean_absolute_error(y_test, lr.predict(X_test)), 2)))
    
    
if st.button('Спрогнозировать курс евро на выбранную дату'):
    predict = df.nPr.reset_index().copy()
    
    if dd == datetime.date(2022, 10, 5):
        predict.loc[236] = ''
        predict.loc[236].dDate = pd.to_datetime('2022-10-05')
        predict.loc[236].nPr = 0
    elif dd == datetime.date(2022, 10, 6):
        predict.loc[236] = ''
        predict.loc[237] = ''
        
        predict.loc[236].dDate = pd.to_datetime('2022-10-05')
        predict.loc[236].nPr = 0
        
        predict.loc[237].dDate = pd.to_datetime('2022-10-06')
        predict.loc[237].nPr = 0
    
    predict.dDate = pd.to_datetime(predict.dDate)
    predict.set_index('dDate', inplace = True)
    
    
    add_features(predict, 40, 3)
    
    
    if dd == datetime.date(2022, 10, 5):
        a = predict.drop('nPr', axis = 1).tail(1).copy()
        st.text('Предсказанный курс евро на 5 октября - ', round(lr.predict(a)[0], 4))
    elif dd == datetime.date(2022, 10, 6):
        a = predict.drop('nPr', axis = 1).tail(2).copy()
        st.text('Предсказанный курс евро на 6 октября - ', round(lr.predict(a)[1], 4))
    

    st.line_chart(predict)
