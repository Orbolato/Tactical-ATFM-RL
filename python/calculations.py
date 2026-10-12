# Filter rows where 'day' is in test_days
filtered_data = data[data['day'].isin(test_days)].reset_index(drop=True)
day = filtered_data.loc[0, 'day']
counter = 0
day_counter = 0

for i in range(len(filtered_data)):
    if filtered_data.loc[i, 'day'] == day:
        filtered_data.loc[i, 'time'] = counter
        filtered_data.loc[i, 'day'] = day_counter
        counter += 1
    else:
        day_counter += 1
        counter = 0
        day = filtered_data.loc[i, 'day']
        filtered_data.loc[i,'day'] = day_counter
        filtered_data.loc[i,'time'] = counter
        counter += 1
filtered_data['day'] = filtered_data['day'].astype(int)
filtered_data['time'] = filtered_data['time'].astype(int)

# Assuming 'data' is your pandas DataFrame
# Extract the required columns
runway_data = filtered_data[['day','time','weather_condition', 'wind_state', 
                    'arrival_demand_40', 'departure_demand']].copy()

# (Optional) Convert columns to appropriate types if needed
# Example: Convert 'time' to a standard format if it's not already
# runway_data['time'] = pd.to_datetime(runway_data['time']).dt.strftime('%H:%M')

# Save to CSV with AMPL-friendly formatting:
# - No index column
# - Using comma as separator (default)
runway_data.to_csv('runway_data.csv', index=False)

print("File 'runway_data.csv' has been created successfully.")

filtered_data_test = data[data['day'].isin(test_days)].reset_index(drop=True)

filtered_data_test = filtered_data_test[['greedy_configuration','greedy_arrivals_40','greedy_departures']].copy()

import runway_optimization_ga as ga

opt_list = []
for day in test_days:
    
    # Initialize the GA optimizer
    optimizer = ga.RunwayOptimizerGA(
        alpha=2.0,  # Weight for arrival queue
        beta=1,
        delta=0.3,  # Weight for configuration switches
        data=data,
        u=u,
        u_max=u_max,
        v=v,
        v_max=v_max,
        sep_min = 5,
        sep_max = 30,
        sep_step = 5,
        possible_actions=possible_actions,
        valid_bool=valid_bool,
        wind_states=wind_states
    )
    
    # Set GA parameters for faster demonstration
    optimizer.population_size = 30
    optimizer.generations = 10000
    
    # Optimize a single day
    day_to_optimize = day
    res = 1e9
    while res == 1e9:
        results = optimizer.optimize_day(day_to_optimize, verbose=True, use_warm_start=True)
        res = results['objective']
    
    # Visualize results
    optimizer.visualize_results(day_to_optimize)
    optimizer.plot_convergence(day_to_optimize)
    opt_rew = -results['objective']
    opt_list.append(opt_rew)

final_opt_list = []
worst_days = []
for i in range(len(test_days)):
    if(greedy_list[i] > opt_list[i]):
        print('Greedy wins for day ' + test_days[i] + '.\n')
        final_opt_list.append(greedy_list[i])
        worst_days.append(test_days[i])
    elif(greedy_list[i] < opt_list[i]):
        print('GA wins for day ' + test_days[i] + '.\n')
        final_opt_list.append(opt_list[i])
    else:
        print('Draw.\n')
        final_opt_list.append(opt_list[i])

plt.figure()
plt.plot(greedy_list)
# plt.plot(opt_list)
plt.plot(final_opt_list)
plt.legend(['Greedy','GA'])
plt.show()

###########################################################

for i in range(len(test_days)):
    if(model_list_1[i] > final_opt_list[i]):
        print('Model wins for day ' + test_days[i] + '.\n')
    elif(model_list_1[i] < final_opt_list[i]):
        print('GA wins for day ' + test_days[i] + '.\n')
    else:
        print('Draw.\n')

plt.figure()
plt.plot(model_list_1)
# plt.plot(opt_list)
plt.plot(final_opt_list)
plt.legend(['Model','GA'])
plt.show()

plt.show()
plt.plot(model_list_2)
plt.plot(model_list_2_eval)
# plt.plot(base_list_2)
plt.plot(np.array(opt_list)[days_2])
plt.xlabel('Days')
plt.ylabel('Total Reward')
# plt.savefig('Results/DB_2/dcb_ppo_db_2_results.eps', format='eps')
plt.show()



################################################

for i in range(len(test_days)):
    if(final_opt_list[i] < opt_list[i]):
        print('CM wins for day ' + test_days[i] + '.\n')
        final_opt_list[i] = opt_list[i]
    elif(final_opt_list[i] > opt_list[i]):
        print('DCB wins for day ' + test_days[i] + '.\n')
    else:
        print('Draw.\n')

plt.figure()
# plt.plot(greedy_list)
plt.plot(opt_list)
plt.plot(final_opt_list)
plt.legend(['CM','DCB'])
plt.show()