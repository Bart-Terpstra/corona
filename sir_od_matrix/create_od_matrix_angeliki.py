import pandas as pd
import numpy as np
import os


# %% Read excel file with the population of North Braband
pop_age = pd.read_excel(r"C:\Users\20195526\Documents\GitHub\corona\sir_od_matrix\data\Bevolking  2019 - Gemeenten Noord-Brabant_1.xlsx", skiprows=2, index_col=0)

# %% Define the working and non-working groups --> actually the ones going out of home and the ones not
pop_age_work_adults = pop_age.loc[:,"age_22.5":"age_62.5"]
pop_age_kids = pop_age.loc[:,"age_2.5":"age_17.5"]

pop_age_work = pop_age_kids.join(pop_age_work_adults)
pop_age_no_work = pop_age.loc[:,"age_67.5":"age_97.5"]

# %% Find the total population per defined group 
pop_age_work['sum_work'] = pop_age_work.sum(axis=1)
pop_age_no_work["sum_no_work"] = pop_age_no_work.sum(axis=1)

# Put the totals in a dataframe 
frame = {'sum_no_work': pop_age_no_work["sum_no_work"], 'sum_work': pop_age_work['sum_work'] } 
pop_age_groups = pd.DataFrame(frame) 

#%% Make some corrections in the municipalities of North Braband
pop_age_groups =  pop_age_groups.rename(index = {'Nuenen c.a.': "Nuenen"})
pop_age_groups.reset_index

# %% Read the od_input data of employees traveling to other cities
od = pd.read_excel(r'C:\Users\20195526\TU Eindhoven\Terpstra, B.W.C. - Case_study_corona\Resources\Live_work_distances_employees_per_municipality_per_year.xlsx')
od.rename(columns={"Woonregio's": "origin", "Werkregio's": "destination", "Banen van werknemers (x 1 000)": "Employees"}, inplace=True)
print('Number of rows: ',len(od))

# %% Take the mean of several years (2015-2019)
od_grouped= od.groupby(['origin', 'destination']).agg({'Employees':['mean']})

od_grouped.columns = [col[0] for col in od_grouped.columns]
od_1 = od_grouped.copy()
od_1.reset_index(inplace=True)

#%% Make some corrections in the municipalities of North Braband in the "Live_work_distances_employees_per_municipality_per_year" file
od_1.replace({'Aalburg':'Altena', 'Werkendam':'Altena', 'Woudrichem':'Altena', 'Maasdonk':"s-Hertogenbosch","'s-Hertogenbosch":"s-Hertogenbosch",
           'Veghel':'Meierijstad','Schijndel':'Meierijstad', 'Sint-Oedenrode':'Meierijstad'},inplace=True)


# Find the unique values of origins & destinations
munic_od_orig = od_1["origin"].sort_values().unique()
munic_od_des = od_1["destination"].sort_values().unique()
len(munic_od_orig), len(munic_od_des), len(pop_age_groups.index)


# %% Add the origin destination combinations that are missing
origin = []
destination = []
employees = []
for i in munic_od_des:
    included = od_1[od_1['destination']==i]['origin'].unique()
    missing = list(set(munic_od_des)-set(included))
    missing.sort()
    for j in missing:
        destination.append(i)
        origin.append(j)
        employees.append(0)
print(missing)
print(origin,destination,employees)

# Add origin that did not exist
add_destinations = pd.DataFrame({'origin':origin, 'destination':destination, 'Employees':employees})

# Merge these additions with the existing OD dataframe
od_1_merged = od_1.append(add_destinations, ignore_index=True)

# Where there are NaN values, replace with 0
od_1_merged["Employees"].fillna(0, inplace=True)
od_1_merged[od_1_merged["destination"] == 'Nuenen']

#%% Create an OD matrix from the dataframe long format
od_pivot = pd.pivot_table(od_1_merged, values='Employees', index=['origin'],
                    columns=['destination'])

# Scale the number of travelers with 1000
od_pivot = od_pivot*1000
od_pivot.to_csv(r'C:\Users\20195526\TU Eindhoven\Terpstra, B.W.C. - Case_study_corona\Resources/travelling_to_other_cities_x1000_ang.csv')

# Create a numpy matrix of its values 
od_matrix = od_pivot.to_numpy()

# %% Create the OD_matrix_w, only for the working population
od_matrix_w = od_matrix.copy()
# Sum of the OD matrix inhabitants that are traveling
travel_sum = od_matrix_w.sum(axis=1)
# Remaining part that is not traveling
remainers_w = pop_age_groups["sum_work"] - travel_sum
# Add remainers to the OD matrix
np.fill_diagonal(od_matrix_w, od_matrix_w.diagonal() + remainers_w)
# Round to integer (remains a float though)
od_matrix_w = np.round(od_matrix_w, 0)
# Create a dataframe with the updated OD matrix values that now include the remainers
df_od_matrix_w = pd.DataFrame(od_matrix_w, columns=od_pivot.columns, index=od_pivot.index)


#%% Create the OD_matrix_y, only for the non-working population
df_od_matrix_w_no = df_od_matrix_w.copy()
# Make everything zero. No travelling for these ages
df_od_matrix_w_no.values[np.arange(df_od_matrix_w_no.shape[0])] = 0
# Create a numpy matrix of its values 
od_matrix_w_no = df_od_matrix_w_no.to_numpy()
# Add the non_working population to the diagonals
np.fill_diagonal(od_matrix_w_no, pop_age_groups["sum_no_work"])


# In[492]:
total_pop_from_od = np.diagonal(od_matrix_w_no)+ pop_age_groups["sum_work"]
total_pop_from_age_pop = pop_age_groups.sum(axis=1)
check = total_pop_from_od - total_pop_from_age_pop

# %% Generate a csv of the df_od_matrix_w
df_od_matrix_w.to_excel(r'C:\Users\20195526\Documents\GitHub\corona\sir_od_matrix\data_ang\od_matrix_working.xlsx')
# Generate a csv of the df_od_matrix_w_no
df_od_matrix_w_no.to_excel(r'C:\Users\20195526\Documents\GitHub\corona\sir_od_matrix\data_ang\od_matrix_no_working.xlsx')


# In[ ]:




