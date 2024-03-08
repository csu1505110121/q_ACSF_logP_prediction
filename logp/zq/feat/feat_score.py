#!/bin/python
# -*- coding:utf-8 -*-

import numpy as np
import pandas as pd
np.set_printoptions(precision=3)

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

# Feature Selection
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression
from sklearn.feature_selection import RFE

from sklearn.ensemble import RandomForestRegressor

def feat_score(data,k=20):
	data = data.loc[:,(data!=0).any(axis=0)]
	namelists = data.columns[1:-1]

	X = data.iloc[:,1:-1].values
	Y = data['logp'].values
	
	scaler = MinMaxScaler(feature_range=(0,1))
	ss = StandardScaler()

	logp_scaled = ss.fit_transform(Y.reshape(-1,1))
	logp_scaled = pd.DataFrame(logp_scaled, columns=['logp_ss'])

	X_scaled = scaler.fit_transform(X)
	X_scaled = pd.DataFrame(X_scaled,columns=namelists)

	X_ss = X_scaled.values
	Y_ss = logp_scaled.values

	ft = SelectKBest(score_func=f_regression, k=k)
	feature_fit = ft.fit(X_ss,Y_ss)
	
	scores_filter = feature_fit.scores_

	scores_sort = np.argsort(-feature_fit.scores_)

	namelist_sort = namelists[scores_sort]
	
	return namelist_sort, scores_filter[scores_sort]


def RF_score(data):
	data = data.loc[:,(data!=0).any(axis=0)]
	namelists = data.columns[1:-1]

	X = data.iloc[:,1:-1].values
	Y = data['logp'].values
	
	scaler = MinMaxScaler(feature_range=(0,1))
	ss = StandardScaler()

	logp_scaled = ss.fit_transform(Y.reshape(-1,1))
	logp_scaled = pd.DataFrame(logp_scaled, columns=['logp_ss'])

	X_scaled = scaler.fit_transform(X)
	X_scaled = pd.DataFrame(X_scaled,columns=namelists)

	X_ss = X_scaled.values
	Y_ss = logp_scaled.values
	
	rf = RandomForestRegressor(n_estimators=1000,random_state=10)
	
	rf.fit(X_ss,Y_ss.ravel())

	rf_gini = rf.feature_importances_
	
	print(rf_gini)
	indices = rf_gini.argsort()[::-1]
	namelist_sort = namelists[indices]

	return namelist_sort, rf_gini[indices]

def RFE_score(data):
	data = data.loc[:,(data!=0).any(axis=0)]
	namelists = data.columns[1:-1]

	X = data.iloc[:,1:-1].values
	Y = data['logp'].values
	
	scaler = MinMaxScaler(feature_range=(0,1))
	ss = StandardScaler()

	logp_scaled = ss.fit_transform(Y.reshape(-1,1))
	logp_scaled = pd.DataFrame(logp_scaled, columns=['logp_ss'])

	X_scaled = scaler.fit_transform(X)
	X_scaled = pd.DataFrame(X_scaled,columns=namelists)

	X_ss = X_scaled.values
	Y_ss = logp_scaled.values
	
	rfr = RandomForestRegressor()
	rfe = RFE(estimator=rfr, n_features_to_select=20)

	rfe.fit(X_ss, Y_ss)
	
	rfe.ranking_
