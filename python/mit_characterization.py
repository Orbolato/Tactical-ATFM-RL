# MIT characterization at SBGR (Section 4.3.3, Figs. 12 and 13 of the paper).
#
# Extracted verbatim from the data-preprocessing section of the training script
# used during the study; only the file paths were made repository-relative.

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import datetime
import os

###############################################################################
# Data preprocessing
###############################################################################

# Work directory
wd = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Repository root

# Importing data preprocessed in R
print('\nLoading METAR and traffic data...')
data = pd.read_csv(os.path.join(wd, 'data', 'data.csv'))
data['time'] = pd.to_datetime(data['time'], format='mixed')
print('Data loaded.\n')

# Introducing ATFM measure information
print('Loading ATFM measure information...')
file_path = os.path.join(wd, 'data', 'atfm', 'CGNA_MEDIDAS_ATFM_20180101_20231121_processed.csv')
df = pd.read_csv(file_path,sep=';')
print('Done.\n')

print('Processing ATFM measure information...')

# Converting columns from date to datetime
df['time_begin'] = pd.to_datetime(df['time_begin'])
df['time_end'] = pd.to_datetime(df['time_end'])

# Extracting date (without time) to the next grouping
df['date'] = df['time_begin'].dt.date

# Calculate duration in minutes
df['duration_min'] = (df['time_end'] - df['time_begin']).dt.total_seconds() / 60

# Extracting the numeric value of the column TMI
df['separation_nm'] = df['tmi_code'].str.extract(r'(\d+)\s*NM')[0]
df['separation_nm'] = pd.to_numeric(df['separation_nm'])

# Filter only SBGR data
df_sbgr = df[df['destination'] == 'SBGR'].copy()
df_sbgr = df_sbgr[df_sbgr['TMI'] == 'MIT']
# df_sbgr = df_sbgr[df_sbgr['year'] >= 2022]
# df_sbgr = df_sbgr[df_sbgr['year'] <= 2023]

# Sort by begin time
df_sbgr = df_sbgr.sort_values('time_begin')

# Create a list of consolidated intervals
intervals = []
current_start = None
current_end = None
current_max_sep = 0

for _, row in df_sbgr.iterrows():
    if current_start is None:
        # Firs interval
        current_start = row['time_begin']
        current_end = row['time_end']
        current_max_sep = row['separation_nm']
    else:
        if row['time_begin'] <= current_end:
            # Overlapping of intervals - extend the current interval
            current_end = max(current_end, row['time_end'])
            current_max_sep = max(current_max_sep, row['separation_nm'])
        else:
            # New interval without overlapping
            intervals.append({
                'begin': current_start,
                'end': current_end,
                'duration_min': (current_end - current_start).total_seconds() / 60,
                'max_separation_nm': current_max_sep,
                'num_origins': len(df_sbgr[(df_sbgr['time_begin'] <= current_end) & 
                               (df_sbgr['time_end'] >= current_start)])
            })
            current_start = row['time_begin']
            current_end = row['time_end']
            current_max_sep = row['separation_nm']

# Add the last interval
if current_start is not None:
    intervals.append({
        'begin': current_start,
        'end': current_end,
        'duration_min': (current_end - current_start).total_seconds() / 60,
        'max_separation_nm': current_max_sep,
        'num_origins': len(df_sbgr[(df_sbgr['time_begin'] <= current_end) & 
                         (df_sbgr['time_end'] >= current_start)])
    })

# Create consolidated DataFrame
df_intervals = pd.DataFrame(intervals)

# Extract data to a new column
df_intervals['date'] = df_intervals['begin'].dt.date

print('Done.\n')

print('Introducing ATFM measure information into data...')
sep_min = 5
sep_max = 30
sep_step = 5
data['base_sep'] = sep_min

def get_mit_info(start, end, intervals_df):
    # Encontrar intervalos que sobrepõem com o período de 15 minutos
    mask = (
        (intervals_df['begin'] <= start) & 
        (intervals_df['end'] >= end)
    )
    overlapping = intervals_df[mask]
    
    if not overlapping.empty:
        # Se houve sobreposição, retornar True e a separação máxima
        return True, overlapping['max_separation_nm'].max()
    else:
        # Se não houve sobreposição, retornar False e NaN
        return False, None
    
for row in range(len(data)):
    mit_applied, sep = get_mit_info(data.loc[row,'time'],
                                    data.loc[row,'time']+datetime.timedelta(minutes=15),
                                    df_intervals)
    if mit_applied:
        data.loc[row,'base_sep'] = sep
print('Done.\n')

df_intervals = df_intervals[df_intervals['end'].dt.year >= 2022]
df_intervals = df_intervals[df_intervals['begin'].dt.year <= 2023]

print('Plotting ATFM measure statistics...')

# Daily compilation
df_diario = df_intervals.groupby('date').agg({
    'duration_min': 'sum',
    'max_separation_nm': 'max',
    'num_origins': 'max'
}).reset_index()

# Filter relevant data
df_diario = df_diario[df_diario['duration_min'] <= 60*24]
data1 = df_diario['duration_min']
data2 = df_diario['max_separation_nm']

# Plotting statistics
plt.figure()
plt.xlabel('Total duration (min) per day')
plt.ylabel('Frequency')
plt.hist(data1, bins=50, color='blue')
plt.grid(axis='y',linestyle='dashed')
plt.savefig('mit_duration.pdf', format='pdf')
plt.show()

plt.figure()
plt.xlabel('Mean separation (NM) per day')
plt.ylabel('Frequency')
plt.hist(data2, bins=50, color='red')
plt.yticks(list(range(0,250,20)))
plt.grid(axis='y',linestyle='dashed')
plt.xticks(list(range(15,35,5)))
plt.savefig('mit_separation.pdf', format='pdf')
plt.show()

print('Done.\n')

print('Removing partial periods...')
# while(data.loc[0,'time'] != data.loc[0,'time'].replace(data.loc[0,'time'][-8:],'06:00:00')):
#     data.drop(0, axis='rows', inplace=True)
#     data.reset_index(drop=True, inplace=True)
# while(data.loc[len(data)-1,'time'] != data.loc[len(data)-1,'time'].replace(data.loc[len(data)-1,'time'][-8:],'05:45:00')):
#     data.drop(len(data)-1, axis='rows', inplace=True)
#     data.reset_index(drop=True, inplace=True)
while(data.loc[0,'time'].hour != 6):
    data.drop(0, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
while(data.loc[len(data)-1,'time'].hour != 5):
    data.drop(len(data)-1, axis='rows', inplace=True)
    data.reset_index(drop=True, inplace=True)
print('Done.\n')

print('Plotting separation...')
plt.figure()
plt.xlabel('Time steps')
plt.ylabel('Separation (NM)')
plt.yticks(list(range(5,35,5)))
plt.grid(axis='y', linestyle='dashed')
plt.plot(list(range(len(data))), data['base_sep'], color='red')

# Set scientific notation for x-axis
plt.gca().xaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
plt.gca().xaxis.get_major_formatter().set_scientific(True)
plt.gca().xaxis.get_major_formatter().set_powerlimits((0, 0))  # Force scientific notation

plt.savefig('base_separation.pdf', format='pdf')
plt.show()
print('Done.\n')
