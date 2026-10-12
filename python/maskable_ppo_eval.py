from sb3_contrib import MaskablePPO
import improved_gym_capoptimizer as cp
import improved_gym_tdcboptimizer as dcb
import improved_gym_tdcboptimizer_specific as dcb_sp
# from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd
from os import listdir
from os.path import isfile, join
import scipy.stats as stats
import random
from itertools import product
import multiprocessing
from sklearn.linear_model import LinearRegression
# import seaborn as sns

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch.distributions import Distribution
Distribution.set_default_validate_args(False)

###############################################################################
# Function for day counter
###############################################################################

def day_of_the_year(day_string):
    
    day_str_split = day_string.split('-')
    year = int(day_str_split[0])
    month = int(day_str_split[1])
    day = int(day_str_split[2])
    
    if year % 4 == 0:
        if year % 100 == 0 and year % 400 != 0:
            month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        
    counter = 0
    for i in range(month - 1):
        counter += month_days[i]
    counter += day - 1
    
    return counter

def previous_day(day_string):
    
    day_str_split = day_string.split('-')
    year = int(day_str_split[0])
    month = int(day_str_split[1])
    day = int(day_str_split[2])
    
    if year % 4 == 0:
        if year % 100 == 0 and year % 400 != 0:
            month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
        else:
            month_days = [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    else:
        month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    
    if day == 1:
        if month == 1:
            prev_year = year - 1
            prev_month = 12
            prev_day = 31
        else:
            prev_year = year
            prev_month = month - 1
            prev_day = month_days[prev_month - 1]
    else:
        prev_year = year
        prev_month = month
        prev_day = day - 1
    
    prev_day_string = str(prev_year) + '-' + str(prev_month) + '-' + str(prev_day)
    return prev_day_string

###############################################################################
# Runway Configuration Capacity Envelope (RCCE)
###############################################################################

def f(x,y,u): # Departures upper bound for the VMC curve
    v_ub = [] # Upper bound of departures v
    for i in range(len(u)): # For each arrival u_i
        for j in range(len(x)-1):  # For each intersecting point x_j
            if x[j] <= u[i] < x[j+1]: # If x_j <= u_i <= x_(j + 1)
                dy = y[j+1] - y[j]
                dx = x[j+1] - x[j]
                if dx == 0:
                    v_ub.append(y[j])
                else:
                    v_ub.append(dy/dx*(u[i] - x[j]) + y[j])
        if u[i] == x[-1]: # If u_i is the final point of the curve
            dy = y[-1] - y[-2]
            dx = x[-1] - x[-2]
            if dx == 0: # Vertical line
                v_ub.append(y[-2])
            else:
                v_ub.append(0)
        if u[i] > x[-1]: # If u_i is after the final point of the curve
            v_ub.append(0)
        if u[i] < 0: # If u_i is before the initial point of the curve
            v_ub.append(0)
    return np.array(v_ub, dtype=np.int64)

###############################################################################
# Data preprocessing
###############################################################################

# Work directory
wd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Repository root

# Importing data preprocessed in R
print('\nLoading METAR and traffic data...')
data = pd.read_csv(os.path.join(wd, 'data', 'data.csv'))
print('Data loaded.\n')

# Importing TAF data preprocessed in R
print('\nLoading TAF data...')
taf_data = pd.read_csv(os.path.join(wd, 'data', 'new_taf_list.csv'))
print('Data loaded.\n')

# Importing standard TAF data preprocessed in R
print('\nLoading standard TAF data...')
std_taf_data = pd.read_csv(os.path.join(wd, 'data', 'std_taf.csv'))
print('Data loaded.\n')

# print('Creating baseline...')
# for i in range(len(data)-1):
#     data.loc[i,'base_arrivals_40'] = data.loc[i+1,'actual_arrivals']
#     data.loc[i,'base_departures'] = data.loc[i+1,'actual_departures']
#     data.loc[i,'base_configuration'] = data.loc[i+1,'configuration']
# print('Done.\n')

print('Removing partial periods...')
while(data.loc[0,'time'] != data.loc[0,'time'].replace(data.loc[0,'time'][-8:],'06:00:00')):
    data.drop(0, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
while(data.loc[len(data)-1,'time'] != data.loc[len(data)-1,'time'].replace(data.loc[len(data)-1,'time'][-8:],'05:45:00')):
    data.drop(len(data)-1, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
print('Done.\n')

print('Determining periods...')
data['day'] = data['time']
for row in range(len(data)):
    time_split = data.loc[row,'time'].split(' ')
    day_string = time_split[0].split('-')
    year = int(day_string[0])
    month = int(day_string[1])
    day = int(day_string[2])
    day_string = str(year) + '-' + str(month) + '-' + str(day)
    if len(time_split) > 1:
        hour = int(time_split[1][0:2])
    else:
        hour = 0
    if hour < 6:
      data.loc[row,'day'] = previous_day(day_string)
    else:
      data.loc[row,'day'] = day_string
print('Done.\n')

# Calculating demand error
data['arrival_error_40'] = data['arrival_demand_40'] - data['scheduled_arrivals_40']
data['arrival_error_40'] = data['arrival_error_40'].astype(int)
data['arrival_error_100'] = data['arrival_demand_100'] - data['scheduled_arrivals_100']
data['arrival_error_100'] = data['arrival_error_100'].astype(int)
data['departure_error'] = data['departure_demand'] - data['scheduled_departures']
data['departure_error'] = data['departure_error'].astype(int)

# Error in function of scheduled arrivals (40 NM)
y1_mean = []
y1_std = []
data1 = []
sched_arr_40 = np.sort(data['scheduled_arrivals_40'].unique())
x = []
for i in sched_arr_40:
    y1 = data.loc[data['scheduled_arrivals_40'] == i, 'arrival_error_40']
    data1.append(y1)
    x.append([i])
    y1_mean.append(np.mean(y1))
    if np.std(y1) != 0:
        y1_std.append(np.std(y1))
    else:
        y1_std.append(np.std(data['arrival_error_40']))
y1_mean = np.array(y1_mean)
y1_std = np.array(y1_std)
arr_mean_40 = y1_mean.copy()
arr_std_40 = y1_std.copy()
model_arr_mean_40 = LinearRegression()
model_arr_mean_40.fit(x, arr_mean_40)
arr_mean_40_pred = model_arr_mean_40.predict(x)
model_arr_std_40 = LinearRegression()
model_arr_std_40.fit(x, arr_std_40)  
arr_std_40_pred = model_arr_std_40.predict(x)
plt.figure(dpi=1200)
plt.plot(sched_arr_40, y1_mean, 'b')
# plt.plot(sched_arr_40, y1_std, 'g')    
plt.plot(sched_arr_40, arr_mean_40_pred, 'r')
plt.xlabel('Scheduled arrivals (40 NM)')
plt.ylabel('Error')
plt.legend(['Mean CI','Regression CI'])
plt.plot(sched_arr_40, y1_mean - y1_std, '--b')
plt.plot(sched_arr_40, y1_mean + y1_std, '--b')
plt.fill_between(sched_arr_40, 
                 y1_mean - y1_std, 
                 y1_mean + y1_std,
                 color='b',
                 alpha=0.2)
plt.plot(sched_arr_40, arr_mean_40_pred - arr_std_40_pred, '--r')
plt.plot(sched_arr_40, arr_mean_40_pred + arr_std_40_pred, '--r')
plt.fill_between(sched_arr_40, 
                 arr_mean_40_pred - arr_std_40_pred, 
                 arr_mean_40_pred + arr_std_40_pred,
                 color='r',
                 alpha=0.2)  
x1 = sched_arr_40
y1 = data['arrival_error_40']
plt.boxplot(data1, positions=x1)
plt.yticks(ticks=np.array(range(np.sort(y1.unique()).min(), np.sort(y1.unique()).max(), 3)))
plt.savefig('arr_variation_40.pdf', format='pdf')
plt.show()

# Error in function of scheduld arrivals (100 NM)
y1_mean = []
y1_std = []
data1 = []
sched_arr_100 = np.sort(data['scheduled_arrivals_100'].unique())
x = []
for i in np.sort(data['scheduled_arrivals_100'].unique()):
    y1 = data.loc[data['scheduled_arrivals_100'] == i, 'arrival_error_100']
    data1.append(y1)
    x.append([i])
    y1_mean.append(np.mean(y1))
    if np.std(y1) != 0:
        y1_std.append(np.std(y1))
    else:
        y1_std.append(np.std(data['arrival_error_100']))
y1_mean = np.array(y1_mean)
y1_std = np.array(y1_std)
arr_mean_100 = y1_mean.copy()
arr_std_100 = y1_std.copy()
model_arr_mean_100 = LinearRegression()
model_arr_mean_100.fit(x, arr_mean_100)
arr_mean_100_pred = model_arr_mean_100.predict(x)
model_arr_std_100 = LinearRegression()
model_arr_std_100.fit(x, arr_std_100)  
arr_std_100_pred = model_arr_std_100.predict(x)
plt.figure(dpi=1200)
plt.plot(sched_arr_100, y1_mean, 'b')    
# plt.plot(sched_arr_100, y1_std, 'g')    
plt.plot(sched_arr_100, arr_mean_100_pred, 'r')  
plt.xlabel('Scheduled arrivals (100 NM)')
plt.ylabel('Error')
plt.legend(['Mean CI','Regression CI'])
plt.plot(sched_arr_100, y1_mean - y1_std, '--b')
plt.plot(sched_arr_100, y1_mean + y1_std, '--b')
plt.fill_between(sched_arr_100, 
                 y1_mean - y1_std, 
                 y1_mean + y1_std,
                 color='b',
                 alpha=0.2)
plt.plot(sched_arr_100, arr_mean_100_pred - arr_std_100_pred, '--r')
plt.plot(sched_arr_100, arr_mean_100_pred + arr_std_100_pred, '--r')
plt.fill_between(sched_arr_100, 
                 arr_mean_100_pred - arr_std_100_pred, 
                 arr_mean_100_pred + arr_std_100_pred,
                 color='r',
                 alpha=0.2)
x1 = np.sort(data['scheduled_arrivals_100'].unique())
y1 = data['arrival_error_100']
plt.boxplot(data1, positions=x1)
plt.yticks(ticks=np.array(range(np.sort(y1.unique()).min(), np.sort(y1.unique()).max(), 3)))
plt.savefig('arr_variation_100.pdf', format='pdf')
plt.show()

# Error in function of scheduled arrivals
y2_mean = []
y2_std = []
data2 = []
sched_dep = np.sort(data['scheduled_departures'].unique())
x = []
for i in np.sort(data['scheduled_departures'].unique()):
    y2 = data.loc[data['scheduled_departures'] == i, 'departure_error']
    data2.append(y2)
    x.append([i])
    y2_mean.append(np.mean(y2))
    if np.std(y2) != 0:
        y2_std.append(np.std(y2))
    else:
        y2_std.append(np.std(data['departure_error']))
y2_mean = np.array(y2_mean)
y2_std = np.array(y2_std)
dep_mean = y2_mean.copy()
dep_std = y2_std.copy()
model_dep_mean = LinearRegression()
model_dep_mean.fit(x, dep_mean)
dep_mean_pred = model_dep_mean.predict(x)
model_dep_std = LinearRegression()
model_dep_std.fit(x, dep_std)  
dep_std_pred = model_dep_std.predict(x)
plt.figure(dpi=1200)
plt.plot(sched_dep, y2_mean, 'b')    
# plt.plot(sched_dep, y2_std, 'g') 
plt.plot(sched_dep, dep_mean_pred, 'r')
plt.xlabel('Scheduled departures')
plt.ylabel('Error')
plt.legend(['Mean CI','Regression CI'])
plt.plot(sched_dep, y2_mean - y2_std, '--b')
plt.plot(sched_dep, y2_mean + y2_std, '--b')
plt.fill_between(sched_dep, 
                 y2_mean - y2_std, 
                 y2_mean + y2_std,
                 color='b',
                 alpha=0.2) 
plt.plot(sched_dep, dep_mean_pred - dep_std_pred, '--r')
plt.plot(sched_dep, dep_mean_pred + dep_std_pred, '--r')
plt.fill_between(sched_dep, 
                 dep_mean_pred - dep_std_pred, 
                 dep_mean_pred + dep_std_pred,
                 color='r',
                 alpha=0.2)     
x2 = np.sort(data['scheduled_departures'].unique())
y2 = data['departure_error']
plt.boxplot(data2, positions=x2)
plt.yticks(ticks=np.array(range(np.sort(y2.unique()).min(), np.sort(y2.unique()).max(), 3)))
plt.savefig('dep_variation.pdf', format='pdf')
plt.show()

# Plotting histograms and normal distributions (40 NM)
y1 = data['arrival_error_40']
plt.figure(dpi=1200)
plt.hist(y1, density=True, bins=np.sort(data['arrival_error_40'].unique()))
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Arrival Error (40 NM)')

# Plotting normal distribution
arr_mu_40 = np.average(data['arrival_error_40'])
arr_sigma_40 = np.std(data['arrival_error_40'])
x = np.linspace(arr_mu_40 - 5*arr_sigma_40, arr_mu_40 + 5*arr_sigma_40, 100)
plt.plot(x, stats.norm.pdf(x, arr_mu_40, arr_sigma_40))
plt.legend(['Normal Distribution','Histogram (40 NM)'])
plt.savefig('arr_dist_40.pdf', format='pdf')
plt.show()

# Plotting histograms and normal distributions (100 NM)
y1 = data['arrival_error_100']
plt.figure(dpi=1200)
plt.hist(y1, density=True, bins=np.sort(data['arrival_error_100'].unique()))
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Arrival Error (100 NM)')

# Plotting normal distribution
arr_mu_100 = np.average(data['arrival_error_100'])
arr_sigma_100 = np.std(data['arrival_error_100'])
x = np.linspace(arr_mu_100 - 5*arr_sigma_100, arr_mu_100 + 5*arr_sigma_100, 100)
plt.plot(x, stats.norm.pdf(x, arr_mu_100, arr_sigma_100))
plt.legend(['Normal Distribution','Histogram (100 NM)'])
plt.savefig('arr_dist_100.pdf', format='pdf')
plt.show()

# Plotting histograms
y2 = data['departure_error']
plt.figure(dpi=1200)
plt.hist(y2, density=True, bins=np.sort(data['departure_error'].unique()))
plt.xlabel('Error')
plt.ylabel('Probability')
plt.title('Departure Error')

# Plotting normal distribution
dep_mu = np.average(data['departure_error'])
dep_sigma = np.std(data['departure_error'])
x = np.linspace(dep_mu - 5*dep_sigma, dep_mu + 5*dep_sigma, 100)
plt.plot(x, stats.norm.pdf(x, dep_mu, dep_sigma))
plt.legend(['Normal Distribution','Histogram'])
plt.savefig('dep_dist.pdf', format='pdf')
plt.show()

# Arrivals and departures columns
while taf_data.loc[0, 'time'] != data.loc[0, 'time']:
    taf_data.drop(0, axis='rows', inplace=True)
    taf_data.reset_index(drop=True, inplace=True)
    std_taf_data.drop(0, axis='rows', inplace=True)
    std_taf_data.reset_index(drop=True, inplace=True)
while taf_data.loc[taf_data.shape[0]-1, 'time'] != data.loc[data.shape[0]-1, 'time']:
    taf_data.drop(taf_data.shape[0]-1, axis='rows', inplace=True)
    taf_data.reset_index(drop=True, inplace=True)
    std_taf_data.drop(std_taf_data.shape[0]-1, axis='rows', inplace=True)
    std_taf_data.reset_index(drop=True, inplace=True)
taf_data['arrival_demand_40_edmus'] = std_taf_data['arrival_demand_40_edmus'] = data['scheduled_arrivals_40_edmus']
taf_data['arrival_demand_40_vunox'] = std_taf_data['arrival_demand_40_vunox'] = data['scheduled_arrivals_40_vunox']
taf_data['arrival_demand_40_zares'] = std_taf_data['arrival_demand_40_zares'] = data['scheduled_arrivals_40_zares']
taf_data['arrival_demand_40_buxuk'] = std_taf_data['arrival_demand_40_buxuk'] = data['scheduled_arrivals_40_buxuk']
taf_data['arrival_demand_40'] = std_taf_data['arrival_demand_40'] = data['scheduled_arrivals_40']
taf_data['arrival_demand_100_edmus'] = std_taf_data['arrival_demand_100_edmus'] = data['scheduled_arrivals_100_edmus']
taf_data['arrival_demand_100_vunox'] = std_taf_data['arrival_demand_100_vunox'] = data['scheduled_arrivals_100_vunox']
taf_data['arrival_demand_100_zares'] = std_taf_data['arrival_demand_100_zares'] = data['scheduled_arrivals_100_zares']
taf_data['arrival_demand_100_buxuk'] = std_taf_data['arrival_demand_100_buxuk'] = data['scheduled_arrivals_100_buxuk']
taf_data['arrival_demand_100'] = std_taf_data['arrival_demand_100'] = data['scheduled_arrivals_100']
taf_data['departure_demand'] = std_taf_data['departure_demand'] = data['scheduled_departures']

###############################################################################
# Encoding categorical features
###############################################################################

# Preprocessing data
print('Encoding runway configurations and weather condition...')
data['weather_condition'] = data['ifr'] + data['lifr']
data['10R_10L'] = np.zeros(len(data))
data['28L_28R'] = np.zeros(len(data))
for i in range(len(data)):
    if data.loc[i,'configuration'] == '10R|10L':
        data.loc[i,'configuration'] = 1 # '10R_10L'
        data.loc[i,'10R_10L'] = 1
    elif data.loc[i,'configuration'] == '28L|28R':
        data.loc[i,'configuration'] = 2 # '28L_28R'
        data.loc[i,'28L_28R'] = 1
    else:
        data.loc[i,'configuration'] = 0 # None
        
    if data.loc[i,'base_configuration'] == '10R|10L':
        data.loc[i,'base_configuration'] = 1 # '10R_10L'
    elif data.loc[i,'base_configuration'] == '28L|28R':
        data.loc[i,'base_configuration'] = 2 # '28L_28R'
    else:
        data.loc[i,'base_configuration'] = 0 # None
print('Done.\n')

# Defining wind states
print('Defining and encoding wind states...')
cw_max = 20 # Maxium crosswind (knots)
tw_max = 5 # Maximum tailwind (knots)
print('Crosswind limit of ' + str(cw_max) + ' knots.')
print('Tailwind limit of ' + str(tw_max) + ' knots.\n')
data['wind_state'] = np.zeros(len(data)) # Column of wind states
data['wind_state'] = pd.to_numeric(data['wind_state'], downcast='integer')
data['10R_10L_OK'] = np.zeros(len(data)) # 0 - not available, 1 - available
data['28L_28R_OK'] = np.zeros(len(data)) # 0 - not available, 1 - available
# 0 - Both available (0 is always for all runways available)
# 1 - 10R|10L available
# 2 - 28L|28R available
# 3 - Both unavailable
print('Case 0 - Both available.')
print('Case 1 - 10R|10L available.')
print('Case 2 - 28L|28R available.')
print('Case 3 - Both unavailable.\n')
for i in range(len(data)):
    if abs(data.loc[i,'crosswind_10']) > cw_max:
        data.loc[i,'wind_state'] = 3
        data.loc[i,'10R_10L_OK'] = 0
        data.loc[i,'28L_28R_OK'] = 0
    elif data.loc[i,'headwind_10'] < -tw_max:
        data.loc[i,'wind_state'] = 2
        data.loc[i,'10R_10L_OK'] = 0
        data.loc[i,'28L_28R_OK'] = 1
    elif data.loc[i,'headwind_10'] > tw_max:
        data.loc[i,'wind_state'] = 1
        data.loc[i,'10R_10L_OK'] = 1
        data.loc[i,'28L_28R_OK'] = 0
    else:
        data.loc[i,'wind_state'] = 0
        data.loc[i,'10R_10L_OK'] = 1
        data.loc[i,'28L_28R_OK'] = 1

for k in data['wind_state'].unique():
    print('Case ' + str(k) + ' has ' 
          + str(len(data.loc[data['wind_state'] == k,]))
          + ' occurences.')
print('')

###############################################################################
# Runway Configuration Capacity Envelopes (RCCEs)
###############################################################################

# Importing capacity envelopes
print('Generating capacity envelopes...')
points = {0: {}, 1: {}, 2: {}} # 0 - VMC, 1 - IMC, 2 - LIMC
legend = {0: {}, 1: {}, 2: {}}
folder = wd + '//curves//'
files = [f for f in listdir(folder) if isfile(join(folder, f))]
for file in files:
    if file[:3] == 'VMC':
        wc = 0
        if file[4:11] == '28L_28R':
            p = 0.99
        else:
            p = 0.99
    else:
        if file[:3] == 'IMC':
            wc = 1
            if file[4:11] == '28L_28R':
                p = 0.95
            else:
                p = 0.95
        else:
            wc = 2
            if file[4:11] == '28L_28R':
                p = 0.9
            else:
                p = 0.9
    if file[-8:] == '{:.4f}'.format(p).replace('0.','') + '.csv':
        name = file.replace('_RCCE_' + '{:.4f}'.format(p).replace('0.','') + '.csv','')
        curve = pd.read_csv(folder + file)
        print(name)
        if name[-7:] == '10R_10L':
            rc = 1
        elif name[-7:] == '28L_28R':
            rc = 2
        points[wc][rc] = curve
        legend[wc][rc] = name
print('Done.\n')

# Plotting capacity envelopes
legends = []
print('Plotting VMC and IMC capacity envelopes...')
plt.figure(dpi=1200)
for wc in points:
    for rc in points[wc]:
        plt.plot(points[wc][rc]['x'],points[wc][rc]['y'])
        plt.title('RCCEs')
        legends.append(legend[wc][rc])
plt.legend(legends)
plt.xlabel('Arrivals')
plt.ylabel('Departures')
plt.savefig('rcces_gru.pdf', format='pdf')
plt.show()
print('Done.\n')

###############################################################################
# Maximum features' values
###############################################################################

# Total number of time slots
N = len(data)

# Calculating accumulated arrivals and departures
print('Determining a list of days...')
accumulated_arrivals = [data.loc[0,'arrival_demand_40']]
accumulated_departures = [data.loc[0,'departure_demand']]
day = data.loc[0,'day']
days = [day]
for i in range(1,N):
    if data.loc[i,'day'] == day:
        accumulated_arrivals[-1] += data.loc[i,'arrival_demand_40']
        accumulated_departures[-1] += data.loc[i,'departure_demand']
    else:
        accumulated_arrivals.append(data.loc[i,'arrival_demand_40'])
        accumulated_departures.append(data.loc[i,'departure_demand'])
        day = data.loc[i,'day']
        days.append(day)
# accumulated_info = pd.DataFrame(np.array([days, accumulated_arrivals, accumulated_departures], dtype=object).T, columns=['day', 'accumulated_arrivals', 'accumulated_departures'])
# a_max = max(accumulated_info['accumulated_arrivals'])
# d_max = max(accumulated_info['accumulated_departures'])
# print('Maximum accumulated arrivals: ' + str(a_max))
# print('Maximum accumulated departures: ' + str(d_max) + '\n')
a_max = 50
d_max = 50
print('Done.\n')

###############################################################################
# Set of actions
###############################################################################

print('Creating curve points for each weather condition and runway configuration...')
A = 1 # '10R_10L'
B = 2 # '28L_28R'
C = 0 # None
configs = [C, A, B] 
weather_conds = [0, 1, 2] # ['VMC', 'IMC', 'LIMC'] # VMC - 0, IMC - 1, LIMC - 2
wind_states = {0: [A,B], 1: [A], 2: [B], 3: [C]}
u = {}
v = {}
u_max = 0
v_max = 0
for wc in weather_conds:
    for rc in configs:
        if rc == C:
            u[wc, rc] = np.array([0])
            v[wc, rc] = np.array([0])
        else:
            u[wc, rc] = np.array([i for i in range(np.max(points[wc][rc]['x']) + 1)])
            v[wc, rc] = np.array(f(np.array(points[wc][rc]['x']),
                                   np.array(points[wc][rc]['y']),
                                   u[wc, rc]), dtype=np.int64)
        if np.max(u[wc, rc]) > u_max:
            u_max = np.max(u[wc, rc])
        if np.max(v[wc, rc]) > v_max:
            v_max = np.max(v[wc, rc])
print('Done.\n')

# All possible actions

sep_min = 5
sep_max = 30
sep_step = 5

print('\nCreating the set of possible actions...')

possible_actions = [] # Possible actions list
for rc in configs:
    if rc == C:
        for sep_var in range(3):
            possible_actions.append([rc, 0, 0, sep_var]) # Append to the list of possible actions
    else:
        for arr in u[0, rc]:
            for dep in range(v[0, rc][arr] + 1):
                for sep_var in range(3):
                    possible_actions.append([rc, arr, dep, sep_var]) # Append to the list of possible actions
                    
print('Done.\n')

# Counting actions

print('Counting lower bound (CM) actions...')

max_state_action_counter = 0

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            state_action_counter += 1
                            # for sep_var in range(1,3): # Lower bound constraint
                            #     state_action_counter += 1
                    state_action_counter += 1
                    # for sep_var in range(1,3): # Lower bound constraint
                    #     state_action_counter += 1
                for sep in range(sep_min + sep_step, sep_max, sep_step):
                    for rc in wind_states[ws]:
                        u_values = u[wc, rc]  # Precompute u values
                        v_values = v[wc, rc]  # Precompute v values
                        arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                        arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                        for arr in range(arr_min, arr_lim): # For each possible arrival count
                            if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                                state_action_counter += 1
                                # for sep_var in range(3):
                                #     state_action_counter += 1
                        state_action_counter += 1
                        # for sep_var in range(3):
                        #     state_action_counter += 1
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            state_action_counter += 1
                            # for sep_var in range(2): # Upper bound constraint
                            #     state_action_counter += 1
                    state_action_counter += 1 
                    # for sep_var in range(2): # Upper bound constraint
                    #     state_action_counter += 1   
                if state_action_counter > max_state_action_counter:
                    max_state_action_counter = state_action_counter
                        
print('Done.\n')

# Determining the set of actions

print('Creating set of lower bound (CM) valid actions...')

keys = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                sep_max + 1,
                max_state_action_counter),
                -1,
                dtype='int')

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            dep = v_values[arr] # Departure service rates are given by the departure capacity
                            action = [rc, arr, dep, 1]
                            keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                            state_action_counter += 1
                            # for sep_var in range(1,3): # Lower boundary constraint
                            #     action = [rc, arr, dep, sep_var]
                            #     keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                            #     state_action_counter += 1
                    dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                    action = [rc, arr_lim, dep, 1]
                    keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                    state_action_counter += 1
                    # for sep_var in range(1,3): # Lower boundary constraint
                    #     action = [rc, arr_lim, dep, sep_var]
                    #     keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                    #     state_action_counter += 1
                for sep in range(sep_min + sep_step, sep_max, sep_step):
                    for rc in wind_states[ws]:
                        u_values = u[wc, rc]  # Precompute u values
                        v_values = v[wc, rc]  # Precompute v values
                        arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                        arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                        for arr in range(arr_min, arr_lim): # For each possible arrival count
                            if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                                dep = v_values[arr] # Departure service rates are given by the departure capacity
                                action = [rc, arr, dep, 1]
                                keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                                state_action_counter += 1
                                # for sep_var in range(3): # Lower bound constraint
                                #     action = [rc, arr, dep, sep_var]
                                #     keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                                #     state_action_counter += 1
                        dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                        action = [rc, arr_lim, dep, 1]
                        keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                        state_action_counter += 1
                        # for sep_var in range(3): # Lower bound constraint
                        #     action = [rc, arr_lim, dep, sep_var]
                        #     keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                        #     state_action_counter += 1
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            dep = v_values[arr] # Departure service rates are given by the departure capacity
                            action = [rc, arr, dep, 1]
                            keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                            state_action_counter += 1
                            # for sep_var in range(2): # Upper bound constraint
                            #     action = [rc, arr, dep, sep_var]
                            #     keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                            #     state_action_counter += 1
                    dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                    action = [rc, arr_lim, dep, 1]
                    keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                    state_action_counter += 1
                    # for sep_var in range(2): # Upper bound constraint
                    #     action = [rc, arr_lim, dep, sep_var]
                    #     keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                    #     state_action_counter += 1

valid_bool_lb = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                sep_max + 1,
                len(possible_actions)),
                False,
                dtype='bool')

for ws in range(keys.shape[0]):
    for wc in range(keys.shape[1]):
        for a_i in range(keys.shape[2]):
            for d_i in range(keys.shape[3]):
                for sep in range(keys.shape[4]):
                    for state_action_counter in range(keys.shape[5]):
                        idx = keys[ws, wc, a_i, d_i, sep, state_action_counter]
                        if idx != -1:
                            valid_bool_lb[ws, wc, a_i, d_i, sep][idx] = True
        
print('All done.\n')

# Counting actions

print('Counting upper bound actions...')

max_state_action_counter = 0

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            state_action_counter += 1
                            # for sep_var in range(1,3): # Lower bound constraint
                            #     state_action_counter += 1
                    state_action_counter += 1
                    # for sep_var in range(1,3): # Lower bound constraint
                    #     state_action_counter += 1
                for sep in range(sep_min + sep_step, sep_max, sep_step):
                    for rc in wind_states[ws]:
                        u_values = u[wc, rc]  # Precompute u values
                        v_values = v[wc, rc]  # Precompute v values
                        arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                        arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                        for arr in range(arr_min, arr_lim): # For each possible arrival count
                            if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                                state_action_counter += 1
                                # for sep_var in range(3):
                                #     state_action_counter += 1
                        state_action_counter += 1
                        # for sep_var in range(3):
                        #     state_action_counter += 1
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            state_action_counter += 1
                            # for sep_var in range(2): # Upper bound constraint
                            #     state_action_counter += 1
                    state_action_counter += 1 
                    # for sep_var in range(2): # Upper bound constraint
                    #     state_action_counter += 1   
                if state_action_counter > max_state_action_counter:
                    max_state_action_counter = state_action_counter
                        
print('Done.\n')

# Determining the set of actions

print('Creating set of upper bound valid actions...')

keys = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                sep_max + 1,
                max_state_action_counter),
                -1,
                dtype='int')

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            dep = v_values[arr] # Departure service rates are given by the departure capacity
                            action = [rc, arr, dep, 2]
                            keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                            state_action_counter += 1
                            # for sep_var in range(1,3): # Lower boundary constraint
                            #     action = [rc, arr, dep, sep_var]
                            #     keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                            #     state_action_counter += 1
                    dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                    action = [rc, arr_lim, dep, 2]
                    keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                    state_action_counter += 1
                    # for sep_var in range(1,3): # Lower boundary constraint
                    #     action = [rc, arr_lim, dep, sep_var]
                    #     keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                    #     state_action_counter += 1
                for sep in range(sep_min + sep_step, sep_max, sep_step):
                    for rc in wind_states[ws]:
                        u_values = u[wc, rc]  # Precompute u values
                        v_values = v[wc, rc]  # Precompute v values
                        arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                        arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                        for arr in range(arr_min, arr_lim): # For each possible arrival count
                            if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                                dep = v_values[arr] # Departure service rates are given by the departure capacity
                                action = [rc, arr, dep, 2]
                                keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                                state_action_counter += 1
                                # for sep_var in range(3): # Lower bound constraint
                                #     action = [rc, arr, dep, sep_var]
                                #     keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                                #     state_action_counter += 1
                        dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                        action = [rc, arr_lim, dep, 2]
                        keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                        state_action_counter += 1
                        # for sep_var in range(3): # Lower bound constraint
                        #     action = [rc, arr_lim, dep, sep_var]
                        #     keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                        #     state_action_counter += 1
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            dep = v_values[arr] # Departure service rates are given by the departure capacity
                            action = [rc, arr, dep, 1]
                            keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                            state_action_counter += 1
                            # for sep_var in range(2): # Upper bound constraint
                            #     action = [rc, arr, dep, sep_var]
                            #     keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                            #     state_action_counter += 1
                    dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                    action = [rc, arr_lim, dep, 1]
                    keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                    state_action_counter += 1
                    # for sep_var in range(2): # Upper bound constraint
                    #     action = [rc, arr_lim, dep, sep_var]
                    #     keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                    #     state_action_counter += 1

valid_bool_ub = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                sep_max + 1,
                len(possible_actions)),
                False,
                dtype='bool')

for ws in range(keys.shape[0]):
    for wc in range(keys.shape[1]):
        for a_i in range(keys.shape[2]):
            for d_i in range(keys.shape[3]):
                for sep in range(keys.shape[4]):
                    for state_action_counter in range(keys.shape[5]):
                        idx = keys[ws, wc, a_i, d_i, sep, state_action_counter]
                        if idx != -1:
                            valid_bool_ub[ws, wc, a_i, d_i, sep][idx] = True
        
print('All done.\n')

# Counting actions

print('Counting all actions...')

max_state_action_counter = 0

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            for sep_var in range(1,3): # Lower bound constraint
                                state_action_counter += 1
                    for sep_var in range(1,3): # Lower bound constraint
                        state_action_counter += 1
                for sep in range(sep_min + sep_step, sep_max, sep_step):
                    for rc in wind_states[ws]:
                        u_values = u[wc, rc]  # Precompute u values
                        v_values = v[wc, rc]  # Precompute v values
                        arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                        arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                        for arr in range(arr_min, arr_lim): # For each possible arrival count
                            if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                                for sep_var in range(3):
                                    state_action_counter += 1
                        for sep_var in range(3):
                            state_action_counter += 1
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            for sep_var in range(2): # Upper bound constraint
                                state_action_counter += 1
                    for sep_var in range(2): # Upper bound constraint
                        state_action_counter += 1   
                if state_action_counter > max_state_action_counter:
                    max_state_action_counter = state_action_counter
                        
print('Done.\n')

# Determining the set of actions

print('Creating set of all valid actions...')

keys = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                sep_max + 1,
                max_state_action_counter),
                -1,
                dtype='int')

for ws in wind_states:
    for wc in weather_conds:
        for a_i in range(u_max + 1):
            for d_i in range(v_max + 1):
                state_action_counter = 0
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            dep = v_values[arr] # Departure service rates are given by the departure capacity
                            for sep_var in range(1,3): # Lower boundary constraint
                                action = [rc, arr, dep, sep_var]
                                keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                                state_action_counter += 1
                    dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                    for sep_var in range(1,3): # Lower boundary constraint
                        action = [rc, arr_lim, dep, sep_var]
                        keys[ws, wc, a_i, d_i, sep_min, state_action_counter] = possible_actions.index(action)
                        state_action_counter += 1
                for sep in range(sep_min + sep_step, sep_max, sep_step):
                    for rc in wind_states[ws]:
                        u_values = u[wc, rc]  # Precompute u values
                        v_values = v[wc, rc]  # Precompute v values
                        arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                        arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                        for arr in range(arr_min, arr_lim): # For each possible arrival count
                            if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                                dep = v_values[arr] # Departure service rates are given by the departure capacity
                                for sep_var in range(3): # Lower bound constraint
                                    action = [rc, arr, dep, sep_var]
                                    keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                                    state_action_counter += 1
                        dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                        for sep_var in range(3): # Lower bound constraint
                            action = [rc, arr_lim, dep, sep_var]
                            keys[ws, wc, a_i, d_i, sep, state_action_counter] = possible_actions.index(action)
                            state_action_counter += 1
                for rc in wind_states[ws]:
                    u_values = u[wc, rc]  # Precompute u values
                    v_values = v[wc, rc]  # Precompute v values
                    arr_lim = min(a_i, u_values[-1]) # Minimum arrivals (discarding clearly suboptimal actions)
                    arr_min = min(u_values[-1] - list(v_values[::-1]).index(v_values[0]), arr_lim) # Maxium arrivals
                    for arr in range(arr_min, arr_lim): # For each possible arrival count
                        if d_i >= v_values[arr]: # If the departure capacity is lower than the departure queue
                            dep = v_values[arr] # Departure service rates are given by the departure capacity
                            for sep_var in range(2): # Upper bound constraint
                                action = [rc, arr, dep, sep_var]
                                keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                                state_action_counter += 1
                    dep = min(d_i, v_values[arr_lim]) # For the maximum arrival count, take the minimum between the departure queue and the capacity
                    for sep_var in range(2): # Upper bound constraint
                        action = [rc, arr_lim, dep, sep_var]
                        keys[ws, wc, a_i, d_i, sep_max, state_action_counter] = possible_actions.index(action)
                        state_action_counter += 1

valid_bool = np.full((len(wind_states), 
                len(weather_conds), 
                u_max + 1,
                v_max + 1,
                sep_max + 1,
                len(possible_actions)),
                False,
                dtype='bool')

for ws in range(keys.shape[0]):
    for wc in range(keys.shape[1]):
        for a_i in range(keys.shape[2]):
            for d_i in range(keys.shape[3]):
                for sep in range(keys.shape[4]):
                    for state_action_counter in range(keys.shape[5]):
                        idx = keys[ws, wc, a_i, d_i, sep, state_action_counter]
                        if idx != -1:
                            valid_bool[ws, wc, a_i, d_i, sep][idx] = True
        
print('All done.\n')

possible_actions = np.array(possible_actions)

###############################################################################
# Determining weather and wind uncertainty
###############################################################################

n_wc = len(weather_conds)
n_ws = len(wind_states)
p_wc_trans = np.zeros((n_wc, n_wc))
p_wc_error = np.zeros((n_wc, n_wc))
p_ws_trans = np.zeros((n_ws, n_ws))
p_ws_error = np.zeros((n_ws, n_ws))
prev_data = data.loc[:(data.shape[0]-2),:]
prev_data.reset_index(drop=True, inplace=True)
curr_data = data.loc[1:,]
curr_data.reset_index(drop=True, inplace=True)
for i in range(prev_data.shape[0]):
    p_wc_trans[prev_data.loc[i,'weather_condition'], curr_data.loc[i,'weather_condition']] += 1
    p_ws_trans[prev_data.loc[i,'wind_state'], curr_data.loc[i,'wind_state']] += 1
for i in range(p_wc_trans.shape[0]):
    p_wc_trans[i] = p_wc_trans[i] / np.sum(p_wc_trans[i])
for i in range(p_ws_trans.shape[0]):
    p_ws_trans[i] = p_ws_trans[i] / np.sum(p_ws_trans[i])
for i in range(taf_data.shape[0]):
    p_wc_error[taf_data.loc[i,'weather_condition'], data.loc[i,'weather_condition']] += 1
    p_ws_error[taf_data.loc[i,'wind_state'], data.loc[i,'wind_state']] += 1
for i in range(p_wc_error.shape[0]):
    p_wc_error[i] = p_wc_error[i] / np.sum(p_wc_error[i])
for i in range(p_ws_error.shape[0]):
    p_ws_error[i] = p_ws_error[i] / np.sum(p_ws_error[i])

# plt.figure(dpi=2400)
plt.matshow(p_wc_trans)
plt.colorbar()
plt.title('Weather transition probability')
plt.savefig('wc_trans.pdf', format='pdf')
plt.show()

# plt.figure(dpi=2400)
plt.matshow(p_ws_trans)
plt.colorbar()
plt.title('Wind transition probability')
plt.savefig('ws_trans.pdf', format='pdf')
plt.show()

# plt.figure(dpi=2400)
plt.matshow(p_wc_error)
plt.colorbar()
plt.title('Weather confusion probability')
plt.savefig('wc_error.pdf', format='pdf')
plt.show()

# plt.figure(dpi=2400)
plt.matshow(p_ws_error)
plt.colorbar()
plt.title('Wind confusion probability')
plt.savefig('ws_error.pdf', format='pdf')
plt.show()

###############################################################################
# Filtering congested data
###############################################################################

print('Filtering congested data...')

arr_dem_max = max(data['arrival_demand_100'])

# Observation low and high
observation_low = [0,0,0,0,0,sep_min,0]
observation_high = [2,a_max,d_max,2,3,sep_max,arr_dem_max]

# Environment for data filtering
data_filter = dcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, 
                           data=data, 
                           training_days=days, 
                           u=u, u_max=u_max, v=v, v_max=v_max,
                           possible_actions=possible_actions,
                           valid_bool=valid_bool,
                           sep_min=sep_min,
                           sep_max=sep_max,
                           sep_step=sep_step,
                           wind_states=wind_states, 
                           observation_low=observation_low, 
                           observation_high=observation_high)

congested_days = [] # Initializing list of congested days

for day in days: # For each day of the entire dataset

    print(day)
    
    # Compute single action following a greedy policy of a naive agent
    state, info = data_filter.reset(day=day) # Test for the same day
    test_reward = 0 # Initialize test reward
    done = False # Not done
    truncated = False # Not truncated
    
    # While day has not ended
    while not (done or truncated):
        
        # Greedy policy of the naive agent
        action = data_filter.base_action()
        
        # Take the action and observe the state and reward
        state, reward, done, truncated, info = data_filter.base_step(action)
        
        # Updating return
        test_reward += reward

    # Store congested days
    if test_reward <= -10:
        congested_days.append(day)
    
# Filtering
data = data[data.day.isin(congested_days)].reset_index(drop=True)

print('Total of ' + str(len(congested_days)) + ' days.\n')

###############################################################################
# Creating datasets
###############################################################################

# Cutting time window and splitting data
# print('Splitting data into training and test datasets...')
# n_days = len(congested_days) # Total number of days to be considered
# split_ratio = 0.7 # Percentage of the dataset for training
# n_days = min(n_days, len(congested_days))
# sampled_days = random.sample(congested_days,k=n_days)
# training_days = random.sample(sampled_days, k=int(split_ratio*n_days))
# test_days = list(np.array(sampled_days)[~np.isin(sampled_days,training_days)])
# test_days_2 = random.sample(test_days,k=int(0.01*n_days))
# training_days_2 = list(np.array(congested_days)[~np.isin(congested_days,test_days_2)])
# print('Dataset split into ' 
#       + str(len(training_days)) + ' days for training and '
#       + str(len(test_days)) + ' days for testing.')


test_days = ['2023-9-10',
 '2023-12-6',
 '2022-9-19',
 '2022-5-29',
 '2023-6-1',
 '2023-6-30',
 '2023-1-29',
 '2023-11-15',
 '2022-8-2',
 '2023-11-1',
 '2023-1-30',
 '2022-8-17',
 '2023-12-7',
 '2023-1-5',
 '2022-3-18',
 '2023-11-25',
 '2023-3-9',
 '2022-10-14',
 '2023-3-7',
 '2022-9-2',
 '2023-10-7',
 '2023-2-6',
 '2022-5-23',
 '2023-3-23',
 '2023-5-19',
 '2022-2-17',
 '2023-5-3',
 '2023-11-4',
 '2022-1-16',
 '2022-7-29',
 '2023-2-1',
 '2023-3-21',
 '2023-8-8',
 '2023-5-24',
 '2022-8-20',
 '2023-8-19',
 '2023-4-28',
 '2023-8-10',
 '2022-8-1',
 '2023-10-28',
 '2022-10-4',
 '2023-2-15',
 '2022-11-24',
 '2022-11-4',
 '2022-10-29',
 '2022-10-30',
 '2022-11-20',
 '2023-8-11',
 '2022-12-7',
 '2023-9-23',
 '2023-6-29',
 '2023-2-7',
 '2022-2-28',
 '2023-6-21',
 '2022-9-13',
 '2022-10-19',
 '2023-9-18',
 '2023-1-1',
 '2022-6-12',
 '2022-1-8',
 '2023-7-9',
 '2023-1-3',
 '2022-12-2',
 '2023-10-18',
 '2022-12-22',
 '2022-12-9',
 '2023-2-10',
 '2023-4-30',
 '2022-11-19',
 '2022-8-6',
 '2023-8-3',
 '2022-12-30',
 '2022-5-10',
 '2023-12-2',
 '2023-7-17',
 '2023-10-27',
 '2022-11-12',
 '2023-8-24',
 '2023-7-21',
 '2022-11-26',
 '2023-11-23',
 '2023-12-5',
 '2023-4-12',
 '2022-7-14',
 '2022-5-27',
 '2023-6-11',
 '2023-5-12',
 '2022-11-10',
 '2022-7-9',
 '2022-5-4',
 '2022-3-22',
 '2023-5-9',
 '2022-10-8',
 '2023-9-1',
 '2023-10-8',
 '2023-11-18',
 '2023-12-13',
 '2023-3-25',
 '2022-10-17',
 '2023-12-11',
 '2022-8-24',
 '2022-6-17',
 '2022-9-26',
 '2022-12-6',
 '2022-11-1',
 '2022-11-3',
 '2023-10-15',
 '2023-9-4',
 '2022-11-16',
 '2023-10-30',
 '2022-3-30',
 '2022-1-6',
 '2023-8-16',
 '2023-3-12',
 '2022-2-16',
 '2023-6-14',
 '2022-8-3',
 '2022-4-17',
 '2022-11-7',
 '2022-10-1',
 '2023-3-31',
 '2022-11-25',
 '2023-9-16',
 '2023-9-15',
 '2023-1-18',
 '2023-12-21',
 '2023-1-12',
 '2023-10-22',
 '2023-6-17',
 '2022-1-4',
 '2023-10-20',
 '2022-10-28',
 '2022-6-4',
 '2022-3-25',
 '2023-4-8',
 '2022-11-22',
 '2022-4-30',
 '2022-3-24',
 '2022-2-26',
 '2022-12-28',
 '2022-4-25',
 '2022-7-15',
 '2023-11-30',
 '2022-3-12',
 '2022-1-28',
 '2023-2-13',
 '2023-5-23',
 '2023-2-9',
 '2022-6-9',
 '2022-5-26',
 '2022-4-23',
 '2023-8-14',
 '2022-11-27',
 '2023-8-6',
 '2022-5-16',
 '2023-12-22',
 '2022-2-11',
 '2022-4-29',
 '2022-5-19',
 '2022-9-17',
 '2022-7-24',
 '2023-7-6',
 '2022-10-23',
 '2022-4-21',
 '2023-9-25',
 '2022-3-11',
 '2022-10-21',
 '2023-11-21',
 '2022-7-2',
 '2023-12-17',
 '2023-1-24',
 '2023-8-5',
 '2022-5-3',
 '2022-12-29',
 '2023-4-26',
 '2022-3-15',
 '2022-5-28',
 '2023-2-28',
 '2022-6-15',
 '2022-6-10',
 '2022-4-14',
 '2023-1-14',
 '2022-1-24',
 '2022-9-30',
 '2022-12-10',
 '2022-1-13',
 '2023-12-12',
 '2022-9-24',
 '2023-3-14',
 '2022-6-13',
 '2023-6-8',
 '2022-4-4',
 '2023-3-13',
 '2022-7-7',
 '2022-1-2',
 '2022-9-6',
 '2022-8-11',
 '2023-2-5',
 '2023-11-8',
 '2023-8-20',
 '2023-5-25',
 '2022-4-1',
 '2023-5-30',
 '2023-5-28',
 '2023-9-22',
 '2023-5-8',
 '2022-9-27',
 '2022-6-29',
 '2023-11-5']

training_days = list(np.array(congested_days)[~np.isin(congested_days,test_days)])

test_days_2 = ['2023-8-5',
 '2023-1-12',
 '2022-8-17',
 '2023-9-15',
 '2022-12-10',
 '2022-12-30',
 '2023-5-12']

training_days_2 = list(np.array(congested_days)[~np.isin(congested_days,test_days_2)])

days_2 = [171, 126, 11, 123, 184, 71, 86]

###############################################################################
# Greedy policy
###############################################################################

# Introucing greedy policy information
greedy_env = dcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, 
                        data=data, 
                        training_days=training_days, 
                        u=u, u_max=u_max, v=v, v_max=v_max, 
                        possible_actions=possible_actions,
                        valid_bool=valid_bool,
                        sep_min=sep_min,
                        sep_max=sep_max,
                        sep_step=sep_step,
                        wind_states=wind_states, 
                        observation_low=observation_low, 
                        observation_high=observation_high)

data['greedy_configuration'] = data['base_configuration']
data['greedy_arrivals_40'] = data['base_arrivals_40']
data['greedy_departures'] = data['base_departures']
greedy_list = []

for day in test_days:
        
    # Greedy policy
    obs, info = greedy_env.reset(day=day)
    greedy_rew = 0
    counter = 0
    terminated = False
    while not terminated:
        action = greedy_env.naive_action()
        action_array = greedy_env.possible_actions[action]
        idx = data[data['day'] == day].iloc[counter].name
        data.loc[idx, 'greedy_configuration'] = action_array[0]
        data.loc[idx, 'greedy_arrivals_40'] = action_array[1]
        data.loc[idx, 'greedy_departures'] = action_array[2]
        obs, reward, terminated, truncated, info = greedy_env.step(action)
        greedy_rew += reward
        counter += 1
    print('Day: ' + day + ', greedy reward: ' + str(round(greedy_rew,1)))
    greedy_list.append(greedy_rew)

###############################################################################
# Running Algorithm
###############################################################################

print('Starting training process...')
start = time.time()
env_1 = dcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, 
                       data=data, 
                       training_days=training_days, 
                       u=u, u_max=u_max, v=v, v_max=v_max, 
                       possible_actions=possible_actions,
                       valid_bool=valid_bool_lb,
                       sep_min=sep_min,
                       sep_max=sep_max,
                       sep_step=sep_step,
                       wind_states=wind_states, 
                       observation_low=observation_low, 
                       observation_high=observation_high)
# del model_1
model_1 = MaskablePPO("MlpPolicy", env_1, gamma=1, verbose=1)
prev_model_1 = MaskablePPO.load(wd + "/models/Results_eval/dcb_ppo_db_1_lb", 
                            env=env_1) # Load model
model_1.set_parameters(prev_model_1.get_parameters())
# model_1.learn(2_000_000)
end = time.time()

print('Training finished in ' + str(round(end-start, 2)) + ' seconds.')

model_1.save("Results_eval/dcb_ppo_db_1_lb") # Save model

print('Testing...\n')

model_list_1 = []
base_list = []

for day in test_days:

    # Compute single action using the model
    obs, info = env_1.reset(day=day)
    model_rew = 0
    terminated = False
    while not terminated:
        
        # Retrieve current action mask
        action_masks = get_action_masks(env_1)
        action_idx, _states = model_1.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env_1.step(action_idx, training=False, evaluate=True)
        model_rew += reward
    
    # Compute single action following the base policy
    obs, info = env_1.reset(day=day)
    base_rew = 0
    terminated = False
    while not terminated:
        action = env_1.base_action()
        obs, reward, terminated, truncated, info = env_1.base_step(action, evaluate=True)
        base_rew += reward
    
    base_list.append(base_rew)
    model_list_1.append(model_rew)
    
    print('Day: ' + str(env_1._get_info()['day']) 
          + ', model reward: ' + str(round(model_rew,2))
          + ', baseline policy reward: ' + str(round(base_rew,2)))

# Plotting results 
plt.show()
plt.plot(model_list_1,'r')
plt.plot(base_list,'b')
plt.legend(['DCB LB (CM)','Baseline'])
plt.xlabel('Days')
plt.ylabel('Total Reward')
plt.savefig('Results_eval/dcb_ppo_db_1_lb_results.pdf', format='pdf')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
improv_list = np.array(model_list_1) - np.array(base_list)
plt.hist(improv_list, bins=50)
plt.ylabel('Frequency')
plt.xlabel('Difference')
plt.title('DCB LB (CM) vs Baseline')
plt.savefig('Results_eval/dcb_ppo_db_1_lb_improvement.pdf', format='pdf')
plt.show()

print('Done.\n')

# Picking the 1% (7 days) most congested days
# lowest_7 = sorted(model_list_1)[:7]
# lowest_7_idx = []
# for i in range(len(lowest_7)):
#     lowest_7_idx.append(model_list_1.index(lowest_7[i]))
lowest_7_idx = list(range(len(model_list_1)))

###############################################################################
# Running Algorithm
###############################################################################

print('Starting training process...')
start = time.time()
env_2 = dcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, 
                       data=data, 
                       training_days=training_days, 
                       u=u, u_max=u_max, v=v, v_max=v_max, 
                       possible_actions=possible_actions,
                       valid_bool=valid_bool_ub,
                       sep_min=sep_min,
                       sep_max=sep_max,
                       sep_step=sep_step,
                       wind_states=wind_states, 
                       observation_low=observation_low, 
                       observation_high=observation_high)
# del model_1
model_2 = MaskablePPO("MlpPolicy", env_2, gamma=1, verbose=1)
prev_model_2 = MaskablePPO.load(wd + "/models/Results_eval/dcb_ppo_db_1_ub", 
                            env=env_2) # Load model
model_2.set_parameters(prev_model_2.get_parameters())
# model_2.learn(1_000_000)
end = time.time()

print('Training finished in ' + str(round(end-start, 2)) + ' seconds.')

model_2.save("Results_eval/dcb_ppo_db_1_ub") # Save model

print('Testing...\n')

model_list_2 = []
# base_list = []

for day in test_days:

    # Compute single action using the model
    obs, info = env_2.reset(day=day)
    model_rew = 0
    terminated = False
    while not terminated:
        
        # Retrieve current action mask
        action_masks = get_action_masks(env_2)
        action_idx, _states = model_2.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env_2.step(action_idx, training=False, evaluate=True)
        model_rew += reward
    
    # Compute single action following the base policy
    obs, info = env_2.reset(day=day)
    base_rew = 0
    terminated = False
    while not terminated:
        action = env_2.base_action()
        obs, reward, terminated, truncated, info = env_2.base_step(action, evaluate=True)
        base_rew += reward
    
    # base_list.append(base_rew)
    model_list_2.append(model_rew)
    
    print('Day: ' + str(env_2._get_info()['day']) 
          + ', model reward: ' + str(round(model_rew,2))
          + ', baseline policy reward: ' + str(round(base_rew,2)))

# Plotting results 
plt.show()
plt.plot(model_list_1,'r')
plt.plot(model_list_2,'g')
plt.plot(base_list,'b')
plt.legend(['DCB LB (CM)','DCB UB','Baseline'])
plt.xlabel('Days')
plt.ylabel('Total Reward')
plt.savefig('Results_eval/dcb_ppo_db_1_ub_results.pdf', format='pdf')
plt.show()

# Plotting results 
plt.show()
plt.plot(model_list_1,'r')
plt.plot(model_list_2,'g')
plt.legend(['DCB LB (CM)','DCB UB'])
plt.xlabel('Days')
plt.ylabel('Total Reward')
plt.savefig('Results_eval/dcb_ppo_db_1_ub_results_2.pdf', format='pdf')
plt.show()

# Filtering CM results
results_1 = list(np.array(model_list_1)[lowest_7_idx])
results_2 = list(np.array(model_list_2)[lowest_7_idx])

# Plotting results
plt.show()
plt.plot(results_1,'r')
plt.plot(results_2, 'g')
plt.legend(['DCB LB (CM)','DCB UB'])
plt.xlabel('Days')
plt.ylabel('Total Reward')
plt.savefig('Results_eval/dcb_ppo_db_1_ub_results_3.pdf', format='pdf')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
improv_list = np.array(model_list_2) - np.array(model_list_1)
plt.hist(improv_list, bins=50)
plt.title('DCB UB vs DCB LB (CM)')
plt.ylabel('Frequency')
plt.xlabel('Difference')
plt.savefig('Results_eval/dcb_ppo_db_1_ub_improvement.pdf', format='pdf')
plt.show()

print('Done.\n')

###############################################################################
# Running Algorithm
###############################################################################

print('Starting training process...')
start = time.time()
env_3 = dcb.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, 
                       data=data, 
                       training_days=training_days, 
                       u=u, u_max=u_max, v=v, v_max=v_max, 
                       possible_actions=possible_actions,
                       valid_bool=valid_bool,
                       sep_min=sep_min,
                       sep_max=sep_max,
                       sep_step=sep_step,
                       wind_states=wind_states, 
                       observation_low=observation_low, 
                       observation_high=observation_high)
# del model_1
model_3 = MaskablePPO("MlpPolicy", env_3, gamma=1, verbose=1)
prev_model_3 = MaskablePPO.load(wd + "/models/Results_eval/dcb_ppo_db_1", 
                            env=env_3) # Load model
model_3.set_parameters(prev_model_3.get_parameters())
# model_3.learn(1_000_000)
end = time.time()

print('Training finished in ' + str(round(end-start, 2)) + ' seconds.')

model_3.save("Results_eval/dcb_ppo_db_1") # Save model

print('Testing...\n')

model_list_3 = []
# base_list = []

for day in test_days:

    # Compute single action using the model
    obs, info = env_3.reset(day=day)
    model_rew = 0
    terminated = False
    while not terminated:
        
        # Retrieve current action mask
        action_masks = get_action_masks(env_3)
        action_idx, _states = model_3.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env_3.step(action_idx, training=False, evaluate=True)
        model_rew += reward
    
    # Compute single action following the base policy
    obs, info = env_3.reset(day=day)
    base_rew = 0
    terminated = False
    while not terminated:
        action = env_3.base_action()
        obs, reward, terminated, truncated, info = env_3.base_step(action, evaluate=True)
        base_rew += reward
    
    # base_list.append(base_rew)
    model_list_3.append(model_rew)
    
    print('Day: ' + str(env_3._get_info()['day']) 
          + ', model reward: ' + str(round(model_rew,2))
          + ', baseline policy reward: ' + str(round(base_rew,2)))

# Plotting results 
plt.show()
plt.plot(model_list_1,'r')
plt.plot(model_list_2,'g')
plt.plot(model_list_3,'m')
plt.plot(base_list,'b')
plt.legend(['DCB LB (CM)','DCB UB','DCB','Baseline'])
plt.xlabel('Days')
plt.ylabel('Total Reward')
plt.savefig('Results_eval/dcb_ppo_db_1_results.pdf', format='pdf')
plt.show()

# Plotting results 
plt.show()
plt.plot(model_list_1,'r')
plt.plot(model_list_2,'g')
plt.plot(model_list_3,'m')
plt.legend(['DCB LB (CM)','DCB UB','DCB'])
plt.xlabel('Days')
plt.ylabel('Total Reward')
plt.savefig('Results_eval/dcb_ppo_db_1_results_2.pdf', format='pdf')
plt.show()

# Filtering CM results
results_1 = list(np.array(model_list_1)[lowest_7_idx])
results_2 = list(np.array(model_list_2)[lowest_7_idx])
results_3 = list(np.array(model_list_3)[lowest_7_idx])

# Plotting results
plt.show()
plt.plot(results_1,'r')
plt.plot(results_2, 'g')
plt.plot(results_3,'m')
plt.legend(['DCB LB (CM)','DCB UB','DCB'])
plt.xlabel('Days')
plt.ylabel('Total Reward')
plt.savefig('Results_eval/dcb_ppo_db_1_results_3.pdf', format='pdf')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
improv_list = np.array(model_list_3) - np.array(model_list_1)
plt.title('DCB vs DCB LB (CM)')
plt.hist(improv_list, bins=50)
plt.ylabel('Frequency')
plt.xlabel('Difference')
plt.savefig('Results_eval/dcb_ppo_db_1_improvement.pdf', format='pdf')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
improv_list = np.array(model_list_3) - np.array(model_list_2)
plt.title('DCB vs DCB UB')
plt.hist(improv_list, bins=50)
plt.ylabel('Frequency')
plt.xlabel('Difference')
plt.savefig('Results_eval/dcb_ppo_db_1_improvement_2.pdf', format='pdf')
plt.show()

print('Done.\n')

###############################################################################
# Running Algorithm
###############################################################################

results_4 = []
x = [[i] for i in range(100)]
arr_mean_40 = model_arr_mean_40.predict(x)
arr_std_40 = 2*model_arr_std_40.predict(x)
arr_mean_100 = model_arr_mean_100.predict(x)
arr_std_100 = 2*model_arr_std_100.predict(x)
dep_mean = model_dep_mean.predict(x)
dep_std = 2*model_dep_std.predict(x)

for day in np.array(test_days)[lowest_7_idx]:
    
    print('Starting training process...')
    start = time.time()
    env_4 = dcb_sp.RunwayEnv(alpha = 2, beta = 1, delta = 0.3, 
                           taf_data=taf_data, std_taf_data=std_taf_data, data=data, 
                           congested_days=congested_days, 
                           u=u, u_max=u_max, v=v, v_max=v_max, 
                           possible_actions=possible_actions,
                           valid_bool=valid_bool,
                           valid_bool_lb=valid_bool_lb,
                           sep_min=sep_min,
                           sep_max=sep_max,
                           sep_step=sep_step,
                           wind_states=wind_states, 
                           observation_low=observation_low, 
                           observation_high=observation_high,
                           arr_mu_40=arr_mu_40, arr_sigma_40=arr_sigma_40, 
                           arr_mu_100=arr_mu_100, arr_sigma_100=arr_sigma_100, 
                           dep_mu=dep_mu, dep_sigma=dep_sigma,
                           arr_mean_40=arr_mean_40, arr_std_40=arr_std_40, 
                           arr_mean_100=arr_mean_100, arr_std_100=arr_std_100, 
                           dep_mean=dep_mean, dep_std=dep_std,
                           p_wc_trans=p_wc_trans, p_ws_trans=p_ws_trans,
                           p_wc_error=p_wc_error, p_ws_error=p_ws_error,
                           chosen_day=day)
    # del model_1
    model_4 = MaskablePPO("MlpPolicy", env_4, gamma=1, verbose=1)
    prev_model_4 = MaskablePPO.load(wd + "/models/Results_eval/dcb_ppo_fb_2_" + day, 
                                env=env_4) # Load model
    model_4.set_parameters(prev_model_4.get_parameters())
    # model_4.learn(100_000)
    end = time.time()
    
    print('Training finished in ' + str(round(end-start, 2)) + ' seconds.')
    
    model_4.save("Results_eval/dcb_ppo_fb_2_" + day) # Save model
    
    print('Testing...\n')
    
    # Compute single action using the model
    obs, info = env_4.reset(training=False)
    model_rew = 0
    terminated = False
    while not terminated:
        
        # Retrieve current action mask
        action_masks = get_action_masks(env_4)
        action_idx, _states = model_4.predict(obs, action_masks=action_masks, deterministic=True)
        obs, reward, terminated, truncated, info = env_4.step(action_idx, training=False, evaluate=True)
        model_rew += reward
    
    # Compute single action following the base policy
    obs, info = env_2.reset(day=day)
    base_rew = 0
    terminated = False
    while not terminated:
        action = env_2.base_action()
        obs, reward, terminated, truncated, info = env_2.base_step(action, evaluate=True)
        base_rew += reward
    
    print('Day: ' + str(env_2._get_info()['day']) 
          + ', model reward: ' + str(round(model_rew,2))
          + ', baseline policy reward: ' + str(round(base_rew,2)))
    
    # Saving results
    results_4.append(model_rew)

# Plotting results 
plt.show()
plt.plot(results_1,'r')
plt.plot(results_2, 'g')
plt.plot(results_3, 'm')
plt.plot(results_4, 'b')
plt.legend(['DCB LB (CM)','DCB UB','DCB','DCB FB'])
plt.xlabel('Days')
plt.ylabel('Total Reward')
plt.savefig('Results_eval/dcb_ppo_db_fb_2_results.pdf', format='pdf')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
improv_list = np.array(results_4) - np.array(results_1)
plt.title('DCB FB vs DCB LB (CM)')
plt.hist(improv_list, bins=50)
plt.ylabel('Frequency')
plt.xlabel('Difference')
plt.savefig('Results_eval/dcb_ppo_db_fb_2_improvement.pdf', format='pdf')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
improv_list = np.array(results_4) - np.array(results_2)
plt.title('DCB FB vs DCB UB')
plt.hist(improv_list, bins=50)
plt.ylabel('Frequency')
plt.xlabel('Difference')
plt.savefig('Results_eval/dcb_ppo_db_fb_2_improvement_2.pdf', format='pdf')
plt.show()

# Plotting improvement
plt.figure(dpi=1200)
improv_list = np.array(results_4) - np.array(results_3)
plt.title('DCB FB vs DCB DB')
plt.hist(improv_list, bins=50)
plt.ylabel('Frequency')
plt.xlabel('Difference')
plt.savefig('Results_eval/dcb_ppo_db_fb_2_improvement_3.pdf', format='pdf')
plt.show()
    
print('Done.\n')

# # Picking the 1% (7 days) most congested days
# lowest = sorted(model_list_1)[:5]
# lowest_idx = []
# for i in range(len(lowest)):
#     lowest_idx.append(model_list_1.index(lowest[i]))
    
# results_1_new = np.array(results_1)[lowest_idx]
# results_2_new = np.array(results_2)[lowest_idx]
# results_3_new = np.array(results_3)[lowest_idx]
# results_4_new = np.array(results_4)[lowest_idx]

# # Plotting results 
# plt.show()
# plt.plot(results_1_new,'r')
# plt.plot(results_2_new, 'g')
# plt.plot(results_3_new, 'm')
# plt.plot(results_4_new, 'b')
# plt.legend(['DCB LB (CM)','DCB UB','DCB','DCB FB'])
# plt.xlabel('Days')
# plt.ylabel('Total Reward')
# # plt.savefig('Results_eval/dcb_ppo_db_fb_2_results.pdf', format='pdf')
# plt.show()

# # Plotting improvement
# plt.figure(dpi=1200)
# improv_list = np.array(results_4_new) - np.array(results_1_new)
# plt.title('DCB FB vs DCB LB (CM)')
# plt.hist(improv_list, bins=50)
# plt.ylabel('Frequency')
# plt.xlabel('Difference')
# # plt.savefig('Results_eval/dcb_ppo_db_fb_2_improvement.pdf', format='pdf')
# plt.show()

# # Plotting improvement
# plt.figure(dpi=1200)
# improv_list = np.array(results_4_new) - np.array(results_2_new)
# plt.title('DCB FB vs DCB UB')
# plt.hist(improv_list, bins=50)
# plt.ylabel('Frequency')
# plt.xlabel('Difference')
# # plt.savefig('Results_eval/dcb_ppo_db_fb_2_improvement_2.pdf', format='pdf')
# plt.show()

# # Plotting improvement
# plt.figure(dpi=1200)
# improv_list = np.array(results_4_new) - np.array(results_3_new)
# plt.title('DCB FB vs DCB DB')
# plt.hist(improv_list, bins=50)
# plt.ylabel('Frequency')
# plt.xlabel('Difference')
# # plt.savefig('Results_eval/dcb_ppo_db_fb_2_improvement_3.pdf', format='pdf')
# plt.show()