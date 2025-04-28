# -*- coding: utf-8 -*-
"""
FUNCTION FOR THE sPMV CALCULATION ACCORDING TO BURATTI ET AL.

"""


  
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime
import glob
import os
from scipy import signal
import math
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import utils

#%%%% sPMV FUNCTION: this funtion compute the sPMV accorign to the month you give as inout. Indeed, the sPMV equation changes accoring to the Iclo, hence to the season.
"""
PAY ATTENTION: 
the function takes as input:
    - indoorT (array)
    - indoor RH (array)
    - sensor (str/int)
    - date (datetime)

pay attention to the Nan values --> The input data should not contain NaN values

"""

def sPMV_calculation(indoorT, indoorRH,  date): #sensor,
    
    # inputcomfortcalc = pd.DataFrame({
    #     'date': date, #replace based on how the retrieved data are defined
    #     'indoorT': indoorT, #replace based on how the retrieved data are defined
    #     'pred_indoorRH': indoorRH, #replace based on how the retrieved data are defined
    #     'sensor': ['nan'] * len(date), #replace based on how the retrieved data are defined
    # })
    
    
    
    indoorES = pd.Series()   
    indoorES = 0.61094 * math.e**((17.625 * (indoorT)) / ((indoorT) + 243.04))
    # Calculate water vapor pressure (pv)
    pv=pd.DataFrame()
    pv['vapour_pressure'] = indoorES*( indoorRH  / 100)
    pv['DATE']= date
    
    months=range(1,13)
    # months=range(12)
    seasonal_spmv=[]
    # i=2
    for i in months:
        ###if you select 0 as month --> reference model will be implemented
        # if i==0:
        #     # simplified PMV with coefficient by Rohles - these coefficients are calculated considering a Iclo=0.6 (=trousers, long-sleeved shirt)
        #     a=0.245
        #     b=0.248
        #     c=6.475
            
        #     pmv=round((a*indoorT+b*pv['vapour_pressure']-c),2)
        #     sPMV=pd.DataFrame()
        #     sPMV['DATE']=date
        #     sPMV['sPMV']=pmv
        #     sPMV['related n_sensor']=sensor
        #     sPMV['applied_model']='gound truth - simplified pmv'
            
        #     seasonal_spmv.append(sPMV)
        if i >= 1 and i <= 3:
            #simplified PMV with coefficient by Buratti et al - w
            a_c=0.1478
            b_c=-0.1371
            c_c=2.5239
            #a_c=0.1383
            #b_c=0.0269
            #c_c=3.0190
            
            pvw1=pv.loc[(date.dt.month== i)]
            
        
            pmv_w1=round((a_c*(indoorT.loc[(date.dt.month== i)])+b_c*pvw1['vapour_pressure']-c_c), 2)
            sPMV_w1=pd.DataFrame()
            sPMV_w1['DATE']=date.loc[(date.dt.month== i)]
            sPMV_w1['sPMV']=pmv_w1.astype(float)
            # sPMV_w1['related n_sensor']=sensor.loc[(date.dt.month== i)]
            sPMV_w1['applied_model']='simplified pmv - w'
    
            print('-------selected model buratti - w--------')
           
            seasonal_spmv.append(sPMV_w1)
    
        else:
            print('outofrange')
        
        if i==12:
            #simplified PMV with coefficient by Buratti et al - w
            a_c=0.1478
            b_c=-0.1371
            c_c=2.5239
            pvw2=pv.loc[(date.dt.month== i)]
            pmv_w2=round((a_c*(indoorT.loc[(date.dt.month== i)])+b_c*pvw2['vapour_pressure']-c_c), 2)
            sPMV_w2=pd.DataFrame()
            sPMV_w2['DATE']=date.loc[(date.dt.month== i)]
            sPMV_w2['sPMV']=pmv_w2.astype(float)
            # sPMV_w2['related n_sensor']=sensor.loc[(date.dt.month== i)]
            sPMV_w2['applied_model']='simplified pmv - w'
    
            print('-------selected model buratti - w--------')
          
            seasonal_spmv.append(sPMV_w2)
        else:
            print('outofrange')
        
    ####if you select from 4 to 5 and from 9 to 11 as month --> buratti et al. model will be implemented for middle seasons
    #middle season
        if i >= 4 and i <= 5 :
            #simplified PMV with coefficient by Buratti et al - m
            a_b=0.1383
            b_b=0.0269
            c_b=3.0190
            pvm1=pv.loc[(date.dt.month== i)]
            pmv_m1=round((a_b*indoorT.loc[(date.dt.month== i)]+b_b*pvm1['vapour_pressure']-c_b), 2)
            sPMV_m1=pd.DataFrame()
            sPMV_m1['DATE']=date.loc[(date.dt.month== i)]
            sPMV_m1['sPMV']=pmv_m1.astype(float)
            # sPMV_m1['related n_sensor']=sensor.loc[(date.dt.month== i)]
            sPMV_m1['applied_model']='simplified pmv - m'
    
            
            print('-------selected model buratti - m--------')
            seasonal_spmv.append(sPMV_m1)
        else:  
                print('outofrange')
                
        if i>=9 and i <= 11:
            #simplified PMV with coefficient by Buratti et al - these coefficients have been computed by considering different Iclo
            #in this case, Iclo[0.51-1.00]
            a_b=0.1383
            b_b=0.0269
            c_b=3.0190
            pvm2=pv.loc[(date.dt.month== i)]
            pmv_m2=round((a_b*indoorT.loc[(date.dt.month== i)]+b_b*pvm2['vapour_pressure']-c_b), 2)
            sPMV_m2=pd.DataFrame()
            sPMV_m2['DATE']=date.loc[(date.dt.month== i)]
            sPMV_m2['sPMV']=pmv_m2.astype(float)
            # sPMV_m2['related n_sensor']=sensor.loc[(date.dt.month==i)]
            sPMV_m2['applied_model']='simplified pmv - m'
    
            print('-------selected model buratti - m--------')
            seasonal_spmv.append(sPMV_m2)
            
        else:        
                print('outofrange')
                
        ####if you select from 8 to 6 as month (WARNING: data from june have been not provided) --> buratti et al. model will be implemented for summer season
        #summer
    # for i in months:
        if i >= 6 and i <= 8   :
            #simplified PMV with coefficient by Buratti et al - these coefficients have been computed by considering different Iclo
            #in this case, Iclo[025-0.50]
            a_a=0.2803
            b_a=0.1717
            c_a=7.1383
            pvs=pv.loc[(date.dt.month== i)]
            pmv_s=round((a_a*(indoorT.loc[(date.dt.month== i)])+b_a*(pvs['vapour_pressure'])-c_a), 2)
            sPMV_s=pd.DataFrame()
            sPMV_s['DATE']=date.loc[(date.dt.month== i)]
            sPMV_s['sPMV']=pmv_s.astype(float)
            # sPMV_s['related n_sensor']=sensor.loc[(date.dt.month==i)]
            sPMV_s['applied_model']='simplified pmv - s'
            print('-------selected model buratti - s--------')
            seasonal_spmv.append(sPMV_s)
    
        else:
                print('outofrange')
        sPMV=pd.concat(seasonal_spmv)
        index_nan = sPMV[sPMV.isna().any(axis=1)].index
        if np.any(index_nan):
            print('NaN values are present. Index: '+ str(index_nan))
        else:
            print('No NaN values have been found.')

    return sPMV





    
                
