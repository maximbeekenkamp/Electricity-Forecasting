"""
This file will contain the code which will process the data and prepare it for the model.

Whilst the project is ongoing this docstring will be updated to reflect the work still to be done.

TODO:
- Model data:
    - historical prices / consumers (Data/sales_revenue.csv)
        - filter for: state-ytd-states, relevant states, total, price (cents/kWh), consumers
        - data: year [0], state [1], # consumers [-2], price (cents/kWh) [-1]
    - historical capacity (Data/generation_monthly.csv)
        - filter for: relevant states and total electrical power industry, 2010 onwards !!!!!REMOVE TOTAL row in energy source!!!!!
        - data: year, month, state, energy source, capacity (MWh)
        - process: total capacity per year per type per state 
        - out: year, state, type, capacity (MWh)
    - ~ 80/20 split
        - 2010-2019: training
        - 2020-2023: testing
- Prediction input data:
    - future capacity (Data/plancapacity_annual.csv)
        - filter for: relevant states
        - release marginal impact prediction per type per state for year 0
          then for every subsequent year add capacity of all types to the previous year's capacity
          and once again calculate the marginal impact of the new capacity per type per state
          output pretty graph
    - need to combine with 2023 testing data to get a starting point for capacity, which we then allow to grow with the planned capacity

sales_revenue.csv price is the feature we're trying to predict for year+1
however sales_revenue.csv consumers is also a feature we're using to predict price

Features:
- year
- state
- consumers
- price
- coal capacity
- geothermal capacity
- hydroelectric conventional capacity
- natural gas capacity
- nuclear capacity
- other capacity
- other biomass capacity
- other gases capacity
- petroleum capacity
- pumped storage capacity
- solar thermal and photovoltaic capacity
- total capacity
- wind capacity
- wood and wood-derived fuels capacity

"""
import numpy as np
import pandas as pd

np.random.seed(1234)

class DataSet:
    def __init__(self, states) -> None:
        self.states = states
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_pred = self.load_data()
    
    def load_data(self):
        """
        Loads the data from the various .csv files, and processes them accordingly.

        Returns:
            Tuple of dfs: Returns the training, testing and prediction dataframes.
        """
        revenues_01_09 = pd.read_csv("../Data/sales_revenue_2001-2009.csv")
        revenues_10_23 = pd.read_csv("../Data/sales_revenue_2010-2023.csv")
        capacities = pd.read_csv("../Data/generation_monthly.csv")
        future_capacities = pd.read_csv("../Data/plancapacity_annual.csv")

        
        y_df = self.y_data(revenues_01_09, revenues_10_23)

        y_test = y_df[y_df['Year'] >= 2020] # 2020-2023, shape (32, 3)
        y_train = y_df[y_df['Year'] < 2020] # 2001-2019, shape (152, 3)

        x_df = self.x_data(capacities, y_df)

        x_test = x_df[x_df['Year'] >= 2020] # 2020-2023, shape (32, 16)
        x_train = x_df[x_df['Year'] < 2020] # 2001-2019, shape (152, 16)

        x_pred = self.pred_data(future_capacities, x_train) # shape (29, 16) but the Customers column is just zeros right now        
        
        return x_train, y_train, x_test, y_test, x_pred

    def y_data(self, revenues_01_09, revenues_10_23):
        """
        Processes the revenue data to remove irrelevant data and merge the two dataframes.
        This data is what becomes the y_train and y_test dataframes.

        Args:
            revenues_01_09 (df): EIA Sales and Revenue data for 2001-2009
            revenues_10_23 (df): EIA Sales and Revenue data for 2010-2023

        Returns:
            df: Output dataframe, what our model is predicting
        """
        
        revenues_01_09 = revenues_01_09[revenues_01_09['State'].isin(self.states)]
        revenues_01_09 = revenues_01_09[['Year', 'State', 'TOTAL Customers', 'TOTAL Price']]
        revenues_01_09 = revenues_01_09.rename(columns={"TOTAL Customers": "Customers", "TOTAL Price": "Price"})
        # NOTE: Customers column is missing data pre-2007, will need to apply the same linear model which calculates 
        # the increase in customers post 2023 to the pre-2007 data

        revenues_10_23 = revenues_10_23[revenues_10_23['State'].isin(self.states)]
        revenues_10_23 = revenues_10_23[['Year', 'State', 'TOTAL Customers', 'TOTAL Price']]
        revenues_10_23 = revenues_10_23.rename(columns={"TOTAL Customers": "Customers", "TOTAL Price": "Price"})

        revenues = pd.concat([revenues_01_09, revenues_10_23])
        revenues = revenues.sort_values(by=['Year', 'State']).reset_index(drop=True)
        revenues['Year'] = revenues['Year'].astype(int)

        # NOTE: currently when there is no data for customers a "." is used, or it is NaN, both need to be replaced with 0
        # this is a hold over from the early missing data, so perhaps we ignore for now
        # revenues['Customers'] = revenues['Customers'].fillna(0)
        # revenues['Customers'] = revenues['Customers'].replace(".", 0)
        # revenues['Customers'] = revenues['Customers'].str.replace(',', '').astype(int)

        return revenues.drop(columns=["Customers"])
    
    def x_data(self, capacities, revenues):
        """
        Processes the capacity data to remove irrelevant data and to structure the data in the correct format for the model.
        Effectively we merge the revenues df (- prices) and capacities df on the Year and State columns. To do this we need 
        to 'rotate' (figuratively) the capacities df such that each Type is a new column for the same year and state. ie 
        previously for AL in 2001 we had: 9 rows, 1 for each type of energy source, this will now become 1 row with 13 additional columns 
        (see the Features list in the docstring / README), if a type of energy source is missing for a year and state, the value should be 0.
        This is then repeated per year and state.

        Args:
            capacities (df): EIA Generation Monthly data for 2001-2023
            revenues (df): Our y_train and y_test data, for this function we care about the Customers column but it's neater to pass the whole df

        Returns:
            df: The input dataframe our model is trained on
        """
        capacities = capacities[capacities['STATE'].isin(self.states)]
        capacities = capacities[capacities['TYPE OF PRODUCER'] == "Total Electric Power Industry"]
        capacities = capacities.drop(columns=["TYPE OF PRODUCER"])
        capacities = capacities.rename(columns={"YEAR": "Year", "STATE": "State", "ENERGY SOURCE": "Type", "GENERATION (Megawatthours)": "Capacity"})

        # currently the capacities dataframe has for every year, state, energy type, a row for each month in the year. this needs to be summed
        # ie for AL in 2001, all the generation for coal in 2001 needs to be summed into 1 row, and then the same for all other energy types, 
        # and then repeated for every year and state. after which the month column can be dropped

        capacities['Capacity'] = capacities['Capacity'].str.replace(',', '').astype(int)
        capacities = capacities.groupby(['Year', 'State', 'Type']).sum().reset_index() 
        capacities = capacities.drop(columns=["MONTH"]) 

        capacities = capacities.pivot_table(index=['Year', 'State'], columns='Type', values='Capacity', fill_value=0).reset_index() 
        capacities['Year'] = capacities['Year'].astype(int)

        return pd.merge(revenues.drop(columns=["Price"]), capacities, on=['Year', 'State'])
    
    def pred_data(self, future_capacities, x_train):
        """
        Processes the future capacity data which the trained model will be deployed on.

        Args:
            future_capacities (df): EIA Planned Capacity data for 2023-2027
            x_train (df): Our training data

        Returns:
            df: The input dataframe our trained model will be deployed on
        """
        
        future_capacities = future_capacities[future_capacities['State Code'].isin(self.states)]
        future_capacities = future_capacities.rename(columns={"Planned Year": "Year", "State Code": "State", "Fuel Source": "Type", "Nameplate Capacity": "Capacity"})
        future_capacities = future_capacities.drop(columns=["Producer Type", "Generators", "Facilities", "Summer Capacity"]) 
        future_capacities['Capacity'] = future_capacities['Capacity'].str.replace(',', '').astype(float)
        future_capacities = future_capacities.pivot_table(index=['Year', 'State'], columns='Type', values='Capacity', fill_value=0).reset_index() 
        
        ## making sure that the pred data has the same shape as the data our model was trained on
        missing_columns = set(x_train.columns) - set(future_capacities.columns)

        for column in missing_columns:
            future_capacities[column] = 0

        future_capacities = future_capacities[x_train.columns]
        missing_columns = set(x_train.columns) - set(future_capacities.columns)

        for column in missing_columns:
            future_capacities[column] = 0

        future_capacities['Total'] = future_capacities.drop(columns=['Year', 'State', 'Customers']).sum(axis=1)
        future_capacities = future_capacities[x_train.columns]


        return future_capacities 
