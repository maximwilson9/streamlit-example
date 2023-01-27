#Import packages
import math
import os
from collections import namedtuple
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import beta
import seaborn as sns


#Set Title
st.title('Winning Probability App for Evaluating Hulu Media Tests')

"""
This app calculates the winning probability of cells in incrementality tests we run with our media vendors.
This methodology is based on Hulu's proprietary winning probability framework.
See [here](https://docs.google.com/presentation/d/1pvENyzIFNAN6yQ3bEEPDUD4TyLhz00GJ_CZcX0WhCGA/edit) for more details or reach out to Max Wilson at maxim.wilson@disneystreaming.com.
"""

st.write('Note that the app can only calculate winning probability for multi-cell tests with a control.')
"""
This is a work in progress and currently only works for Facebook.
"""
st.subheader('Test Parameters')
##insert a button with the options for vendors
test_type = st.radio('Select Vendor from the options below:',('Facebook','YouTube','Snapchat','The Trade Desk','Gemini','Roku'))
##if the user clicks facebook, run the code below
##if the user clicks any of the other vendors, print 'coming soon'
if test_type != 'Facebook':
    st.write('Functionality for that vendor is coming soon.')

##add a button that allows the user to upload the test data as a csv or an excel file
##if the user clicks the button, open up a dialog box that allows the user to browse their computer and select the file
##also allow the user to drag and drop the file into the app
fb_file = st.file_uploader('Upload Test Data as a csv or an excel file:', type=['csv', 'xlsx'])


#Read in data from the file the user uploaded
##if the user uploaded an excel file, read the file into a dataframe called raw_fb
##if the user uploaded a csv file, read the file into a dataframe called raw_fb
##if the user uploaded a file that is not a csv or an excel file, print 'please upload a csv or an excel file'
if test_type == 'Facebook':
    if fb_file is not None:
        if fb_file.name.endswith('.xlsx'):
            raw_fb = pd.read_excel(fb_file,header=None, sheet_name='conversion metrics')
        elif fb_file.name.endswith('.csv'):
            raw_fb = pd.read_csv(fb_file,header=None)
        else:
            st.write('Please upload a csv or an excel file.')
else:
    st.write('We only support Facebook right now.')

##add a slider that allows the user to selet the number of simulations to run
##the key for the slider is num_simulations
##the default value is 25000
##the minimum value is 1000 and the maximum value is 100000
##display the number format as an integer with commas

n_sims = st.slider('Select the number of simulations to run (more simulations will give more stable results, but takes longer):', 1000, 100000, 5000)



#pivot data
df = raw_fb.T
df.columns = df.iloc[0]
df= df.drop(df.index[0])
df = pd.DataFrame(df)






##prepare the data
##set column values to be numberic for columns spend, conversions_incremental, conversions_incremental_upper, conversions_confidence, conversions_test, conversions_control_scaled, population_test, population_control
df['spend'] = pd.to_numeric(df['spend'])
df['conversions_incremental'] = pd.to_numeric(df['conversions_incremental'])
df['conversions_incremental_upper'] = pd.to_numeric(df['conversions_incremental_upper'])
df['conversions_confidence'] = pd.to_numeric(df['conversions_confidence'])
df['conversions_test'] = pd.to_numeric(df['conversions_test'])
df['conversions_control_scaled'] = pd.to_numeric(df['conversions_control_scaled'])
df['population_test'] = pd.to_numeric(df['population_test'])
df['population_control'] = pd.to_numeric(df['population_control'])


### in the df dataframe, if the conversions incremental is 0, change it to equal to -1
df['conversions_incremental'] = df['conversions_incremental'].replace(0,-1)


###map FB names to Google template --> Redundant - TO FIX later
df['cpis'] = df['spend']/df['conversions_incremental']
df['Test_name']=df['study_name']
df['conversion_segment']=df['objective_name']
df['study_name']=df['cell_name']
df['absolute_lift']=df['conversions_incremental']
df['absolutelift_CI']=(df['conversions_incremental_upper']-df['conversions_incremental'])/2
df['treatment_user_count']=df['population_test']
df['control_user_count']=df['population_control']
df['treatment_conversions']=df['conversions_test']
df['control_conversions']=df['conversions_control_scaled']*df['population_control']/df['population_test']
df['experiment_cost_usd']=df['spend']
df['absolute_lift_confidence_level']=df['conversions_confidence']

# FIX --> just used to match Youtube for now
df['analysis_date']='2022-05-19'

##add buttons to the app that allows the user to select one or more items the list of available conversion metrics
## The conversion metrics are the unique values in the objective_name column in df
## The conversion metrics are sorted in alphabetical order
## allow the user to select more than one conversion metric
## the key for the buttons is conversion_metrics
## default is all that none are selected
conversion_metrics = st.multiselect('Select conversion metrics to be analyzed:', sorted(df['objective_name'].unique()))
#conversion_metrics = st.multiselect('Select conversion metrics to be analyzed:', sorted(df['objective_name'].unique()), sorted(df['objective_name'].unique()))


##add a subheader to the app that says 'Run Analysis'
st.subheader('Run Analysis')

#if conversion_metrics, test_type, and fb_file are not empty, allow the user to click a button that says 'Run Analysis'
##otherwise, print 'Please select a conversion metric, test vendor, and upload a file'
##only run the code below if the user clicks the 'Run Analysis' button
if conversion_metrics and test_type and fb_file is not None:
    if st.button('Run Analysis'): st.markdown('---')

    else: st.stop()
else:
    st.write('Please select a conversion metric, test vendor, and upload a file.')
    ##stop the code from running below to prevent errors from displaying
    st.stop()



##create a variable test_name that is the value of the first row in the study_name column in df
test_name = df['Test_name'][1]
##write test_name to the app as a subheader
st.subheader('Test Name')
st.write(test_name)

###create a dataframe with topline metrics for the test
##call the dataframe topline_metrics
### The first column is the metric name, the second column is the metric value
##The metrics are: cells, denoted Cells, with the value of the number of cells in the test, which is the count of the unique number of values in the cell_name column in df rounded
##The next metric is spend, denoted Spend (USD) with the value of the sum of the spend column in df for the objective_name column in df that equals 'AddOnPurchase_Disney+ Add-On'
##The next metric is impressions, denoted Impressions with the value of the sum of the impressions column in df for the objective_name column in df that equals 'AddOnPurchase_Disney+ Add-On'
##The next metric is reach, denoted Reach with the value of the sum of the population_reached column in df for the objective_name column in df that equals 'AddOnPurchase_Disney+ Add-On'
topline_metrics = pd.DataFrame({'Metric Name': ['Number of Cells', 'Spend (USD)', 'Impressions', 'Reach'], 'Metric Value': [df['cell_name'].nunique(), df['spend'][df['objective_name']=='AddOnPurchase_Disney+ Add-On'].sum(), df['impressions'][df['objective_name']=='AddOnPurchase_Disney+ Add-On'].sum(), df['population_reached'][df['objective_name']=='AddOnPurchase_Disney+ Add-On'].sum()]})
## round down the metric values in topline_metrics to the nearest integer
topline_metrics['Metric Value'] = topline_metrics['Metric Value'].astype(int)
##write the topline metrics to the app as a subheader
st.subheader('Media Metrics')

# CSS to inject contained in a string
hide_table_row_index = """
            <style>
            thead tr th:first-child {display:none}
            tbody th {display:none}
            </style>
            """

# Inject CSS with Markdown
st.markdown(hide_table_row_index, unsafe_allow_html=True)
##For values above 1000, add a comma to the value
topline_metrics['Metric Value'] = topline_metrics['Metric Value'].apply(lambda x: '{:,}'.format(x))
##For the only the spend row, add a dollar sign to the metric value
topline_metrics['Metric Value'][1] = '$' + topline_metrics['Metric Value'][1]

##write the topline metrics to the app as a table
st.table(topline_metrics)





## reduce df to only the rows where the value of conversion_segment is in the list conversion_metrics
## write a function that returns True if the value of conversion_segment is in the list conversion_metrics
def conversion_segment_filter(x):
    if x in conversion_metrics:
        return True
    else:
        return False
## apply the conversion_segment_filter function to the conversion_segment column in df
df = df[df['conversion_segment'].apply(conversion_segment_filter)]





## Reduce to only useful columns
useful_columns = ['analysis_date',
                  'study_name',
                  'treatment_user_count',
                  'control_user_count',
                  'treatment_conversions',
                  'control_conversions', 'experiment_cost_usd',
                  'cpis',
                  'conversion_segment',
                  'absolute_lift_confidence_level'
                 ]

metrics_df = df[useful_columns]

## reduce metrics_df to only the rows where the value of conversion_segment is in the list conversion_metrics
## write a function that returns True if the value of conversion_segment is in the list conversion_metrics
def conversion_segment_filter(x):
    if x in conversion_metrics:
        return True
    else:
        return False
## apply the conversion_segment_filter function to the conversion_segment column in metrics_df
metrics_df = metrics_df[metrics_df['conversion_segment'].apply(conversion_segment_filter)]

## set seed
np.random.seed(seed = 1234)

alpha_prior = 1e-9
beta_prior = 1e-9

## create a dataframe to hold the results
results = pd.DataFrame()
inc = 0
for objective in metrics_df['conversion_segment'].unique():
    #print(objective)
    for cell in metrics_df[(metrics_df['conversion_segment']==objective)]['study_name'].unique():
        sub_df =  metrics_df[(metrics_df['conversion_segment']==objective) & 
                             (metrics_df['study_name']==cell)]
        sub_df = sub_df.reset_index()
        for dt in metrics_df['analysis_date'].unique():
            sub_df_sub = sub_df[(sub_df['analysis_date']== dt)].reset_index()
            test_alpha_posterior = alpha_prior + sub_df_sub['treatment_conversions'][0]
            #test_beta_posterior = beta_prior + sub_df['population.test'][0] - test_alpha_posterior
            test_beta_posterior = beta_prior + sub_df_sub['treatment_user_count'][0] - sub_df_sub['treatment_conversions'][0]

            ctl_alpha_posterior = alpha_prior + sub_df_sub['control_conversions'][0]
            #ctl_beta_posterior = beta_prior + sub_df['population.control'][0] - ctl_alpha_posterior
            ctl_beta_posterior = beta_prior + sub_df_sub['control_user_count'][0] - sub_df_sub['control_conversions'][0]
            
            
            pdiff = test_alpha_posterior/(test_alpha_posterior+test_beta_posterior) - ctl_alpha_posterior/(ctl_alpha_posterior+ctl_beta_posterior)

    
            temp_df = pd.DataFrame({'dt': dt,
                                    'cell': cell,
                                    #'metric': metric[:-1],
                                    'metric': objective,
                                    'test_alpha_posterior': test_alpha_posterior ,
                                    'test_beta_posterior': test_beta_posterior , 
                                    'ctl_alpha_posterior': ctl_alpha_posterior ,
                                    'ctl_beta_posterior': ctl_beta_posterior ,
                                    'cpis': sub_df_sub['cpis'][0],
                                    'spend': sub_df_sub['experiment_cost_usd'][0],
                                    'population_test':sub_df_sub['treatment_user_count'][0],
                                    'conf_level': sub_df['absolute_lift_confidence_level'][0],
                                    'pdiff': pdiff
                                   }, 
                                    index = [inc])

            results = results.append(temp_df)
            inc+= 1


win_prob_df = pd.DataFrame()
samples_df = pd.DataFrame()

for dt in results['dt'].unique():
    print(dt)
    for metric in results['metric'].unique():
        #print(metric)
        sub_results = results[(results['dt']==dt) & 
                              (results['metric']==metric)].reset_index()
        sims = pd.DataFrame()
        #sims2 = pd.DataFrame()
        wins = np.zeros(len(sub_results['cell'].unique(), ))
        #wins2 = np.zeros(len(sub_results['cell'].unique(), ))
        for i in range(len(sub_results['cell'].unique())):
            if sub_results['conf_level'][i] > 0.0:
                test = beta(sub_results['test_alpha_posterior'][i], sub_results['test_beta_posterior'][i]).rvs(n_sims)
                ctl = beta(sub_results['ctl_alpha_posterior'][i], sub_results['ctl_beta_posterior'][i]).rvs(n_sims)
            else:
                test = np.zeros(n_sims)
                ctl = np.zeros(n_sims)
            sub_samples = pd.DataFrame({'date': dt, 
                                        'cell': sub_results['cell'][i],
                                       'metric': metric, 
                                        'spend': sub_results['spend'][i],
                                        'population_test': sub_results['population_test'][i],
                                        'test_samples': test, 
                                        'control_samples': ctl})
            samples_df = samples_df.append(sub_samples)
            sims[i] = test - ctl
            #sims2[i] = sims[i] * sub_results['population_test'][i]
            wins[i] = 0
            #wins2[i] = 0
            
        for i in range(n_sims):
            wins[np.argmax(sims.loc[i,:])] += 1
            #wins2[np.argmax(sims2.loc[i,:])] += 1
        sub_results['win_prob'] = [(x + 0.0)/n_sims for x in wins]
        #sub_results['win_prob2'] = [(x + 0.0)/n_sims for x in wins2]
        win_prob_df = win_prob_df.append(sub_results)


samples_df['lift_samples'] = samples_df['test_samples'] -samples_df['control_samples']
samples_df['cpis_samples'] = samples_df['spend']/(samples_df['population_test'] * samples_df['lift_samples'] )


###add subheader that displays topline results
st.subheader('Winning Probability and CPiS by Cell')
summary_table = win_prob_df[['dt', 'cell', 'metric', 'win_prob', 'cpis']]
##change the name of the column win_prob to Winning Probability
summary_table = summary_table.rename(columns = {'win_prob': 'Winning Probability'})
##change the name of the column cpis to CPiS
summary_table = summary_table.rename(columns = {'cpis': 'CPiS'})
##Add a table to the app that displays the summary results
##The table has the number of rows equal to the number of unique values metric column of summary_table
##The table has the same number of columns as the twice the number of unique values in the cell column of summary_table
##The first set of columns are named as the concatenation of the unique values of the cell column of summary_table and the string ' CPiS'
##The second set of columns are named as the concatenation of the unique values of the cell column of summary_table and the string ' Win Probability'
##The first set of columns are the values of the cpis column of summary_table for the corresponding metric and cell
##The second set of columns are the values of the win_prob column of summary_table for the corresponding metric and cell
##The table is sorted by the values of the metric column of summary_table in ascending order
to_view = summary_table.pivot_table(index = 'metric', columns = 'cell', values = ['Winning Probability', 'CPiS'])
## format the values in the win_prob columns in to_view to be percentages with one decimal place
to_view['Winning Probability'] = to_view['Winning Probability'].applymap(lambda x: '{:.1%}'.format(x))
##format the values in the cpis columns in to_view to be integers with a dollar sign and commas
to_view['CPiS'] = to_view['CPiS'].applymap(lambda x: '${:,.0f}'.format(x))

## add the table to the app
st.dataframe(to_view)

##The table is sorted by cpis in ascending order
##The table only has one row for each unique value in the cell_name column in df
##The table has three columns: cell_name, conversions_incremental, and cpis
##The value of conversions_incremental is the value of the conversions_incremental column in df for 
##cpis_by_cell = pd.DataFrame({'cell_name': df['cell_name'], 'conversions_incremental': df['conversions_incremental'], 'cpis': df['cpis']})
##round the cpis values to the nearest dollar
##cpis_by_cell['cpis'] = cpis_by_cell['cpis'].astype(int)
##write the cpis_by_cell table to the app
##st.table(cpis_by_cell)







###add subheader for the confidence intervals
st.subheader('Confidence Intervals')

#plot confidence intervals --> Copied directly from Google
sns.set_style('darkgrid')

# CIs for incremental conversions/absolute lift
for obj in df['conversion_segment'].unique():
    sub_df = df[(df['conversion_segment'])==obj]
    # Set sns plot style back to 'poster' to make bars wide on plot
    sns.set_context("poster")

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.tick_params(axis='x', rotation=45, labelsize=12, pad=-10)
    ax.tick_params(axis='y',labelsize=13)

    plt.errorbar(x=np.array(sub_df['study_name']), 
                 y=np.array(sub_df['absolute_lift']), 
                 #yerr=np.array(list(zip(sub_df['absolutelift']-sub_df['absolutelift_CImin'],sub_df['absolutelift_CImax']-sub_df['absolute_lift']))).T,
                 yerr=np.array(sub_df['absolutelift_CI']),
                 fmt='o', ecolor='g', capthick=2)

    # Set title & labels
    ##plt.title('Incremental Buyers w/ 90% Confidence Intervals {y}',fontsize=15)
    plt.title(obj,fontsize=15)
    ax.set_ylabel("Incremental Conversions",fontsize=12)
    ax.set_xlabel('', fontsize=12)


    # Line to define zero on the y-axis
    ax.axhline(y=0, linestyle='--', color='red', linewidth=1)
    
    ## set density number format to include comma separator
    current_values = plt.gca().get_yticks()
    plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])

    plt.show()
    st.write(fig)



## Add density plots
st.subheader('Density Plots')
#### In a column

sns.set()
sns.set_context('talk')
sns.set_style('darkgrid')


grid = sns.FacetGrid(samples_df[['date','cell','metric','lift_samples']], 
                     row='metric',
                     #row='date',
                     hue='cell' ,height=3.5, aspect=2)

grid.map(sns.kdeplot, 'lift_samples', shade=True)
#grid.set(xlim=(-100000, 100000))


## Set Titles
grid.set_titles(row_template="{row_name}")

## Add Legend
grid.add_legend(title="Cell")

### Set Axis Labels
grid.set_axis_labels("Absolute Lift", "Density")


## set density number format to include comma separator
current_values = plt.gca().get_yticks()
plt.gca().set_yticklabels(['{:,.0f}'.format(x) for x in current_values])
xx_values = plt.gca().get_xticks()
##plt.gca().set_xticklabels(['{:%.0f%%}'.format(x) for x in xx_values])

fig2 = grid.fig
## write to streamlit
st.pyplot(fig2)
