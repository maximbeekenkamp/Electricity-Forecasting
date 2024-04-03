import numpy as np
import pandas as pd
import tensorflow as tf

from consumer_growth import ConsumerGrowth

class DataSet:
    def __init__(self, states, batch_size, test_batch_size, tf_data_type, modeltype="linear"):
        self.states = states
        self.modeltype = modeltype
        self.bs = batch_size
        self.tsbs = test_batch_size
        self.tf_data_type = tf_data_type
        self.x_train, self.y_train, self.x_test, self.y_test, self.x_pred = self.load_data()

    def load_data(self):
        """
        Loads the data from the various .csv files, and processes them accordingly.

        Returns:
            Tuple of dfs: Returns the training, testing and prediction dataframes.
        """
        revenues_01_09 = pd.read_csv("Data/sales_revenue_2001-2009.csv")
        revenues_10_23 = pd.read_csv("Data/sales_revenue_2010-2023.csv")
        capacities = pd.read_csv("Data/generation_monthly.csv")
        future_capacities = pd.read_csv("Data/plancapacity_annual.csv")

        CG = self.y_data(revenues_01_09, revenues_10_23)
        revenues = CG.df
        revenues = self.makeFloat(revenues)

        y_df = revenues.drop(columns=["Customers", "State"])
        self.summary_statistics(y_df, "Y")

        y_test = y_df[y_df["Year"] >= 2020]  # 2020-2023, shape (32, 2)
        y_test = y_test.drop(columns=["Year"])
        y_train = y_df[y_df["Year"] < 2020]  # 2001-2019, shape (152, 2)
        y_train = y_train.drop(columns=["Year"])

        x_df = self.x_data(capacities, revenues)
        self.summary_statistics(x_df, "X")

        x_test = x_df[x_df["Year"] >= 2020]  # 2020-2023, shape (32, 16)
        print(x_test.head())
        x_train = x_df[x_df["Year"] < 2020]  # 2001-2019, shape (152, 16)

        x_pred = self.pred_data(future_capacities, x_train, CG)  # shape (29, 16) but the Customers column is just zeros right now
        self.summary_statistics(x_pred, "X Pred")

        x_train = self.makeTensor(x_train) # shape (152, 23)
        y_train = self.makeTensor_Y(y_train) # shape (152, 1)
        x_test = self.makeTensor(x_test) # shape (32, 23)
        y_test = self.makeTensor_Y(y_test) # shape (32, 1)
        x_pred = self.makeTensor(x_pred) # shape (29, 23)

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
        Columns = ["Year", "State", "TOTAL Customers", "TOTAL Price"]

        revenues_01_09 = revenues_01_09[revenues_01_09["State"].isin(self.states)]
        revenues_01_09 = revenues_01_09[Columns]
        revenues_01_09 = revenues_01_09.rename(
            columns={"TOTAL Customers": "Customers", "TOTAL Price": "Price"}
        )

        revenues_10_23 = revenues_10_23[revenues_10_23["State"].isin(self.states)]
        revenues_10_23 = revenues_10_23[Columns]
        revenues_10_23 = revenues_10_23.rename(
            columns={"TOTAL Customers": "Customers", "TOTAL Price": "Price"}
        )

        revenues = pd.concat([revenues_01_09, revenues_10_23])
        revenues = revenues.sort_values(by=["Year", "State"]).reset_index(drop=True)
        revenues["Year"] = revenues["Year"].astype(int)

        revenues["Customers"] = revenues["Customers"].replace(".", 0)
        revenues["Customers"] = revenues["Customers"].str.replace(",", "").astype(float)

        revenues = self.OneHotEncode(revenues)
        
        return ConsumerGrowth(revenues, self.modeltype)

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
        capacities = capacities[capacities["STATE"].isin(self.states)]
        capacities = capacities[
            capacities["TYPE OF PRODUCER"] == "Total Electric Power Industry"
        ]
        capacities = capacities.drop(columns=["TYPE OF PRODUCER"])
        capacities = capacities.rename(
            columns={
                "YEAR": "Year",
                "STATE": "State",
                "ENERGY SOURCE": "Type",
                "GENERATION (Megawatthours)": "Capacity",
            }
        )
        capacities = self.OneHotEncode(capacities)
        capacities["Capacity"] = capacities["Capacity"].str.replace(",", "").astype(int)
        capacities = capacities.groupby(["Year", "State", "Type"]).sum().reset_index()
        capacities = capacities.drop(columns=["MONTH"])

        capacities = capacities.pivot_table(index=["Year", "State"], columns="Type", values="Capacity", fill_value=0).reset_index()
        capacities["Year"] = capacities["Year"].astype(int)
        
        return self.makeFloat(
            pd.merge(
                revenues.drop(columns=["Price"]), capacities, on=["Year", "State"]
                )
            )

    def pred_data(self, future_capacities, x_train, modelObject):
        """
        Processes the future capacity data which the trained model will be deployed on.

        Args:
            future_capacities (df): EIA Planned Capacity data for 2023-2027
            x_train (df): Our training data

        Returns:
            df: The input dataframe our trained model will be deployed on
        """

        future_capacities = future_capacities[future_capacities["State Code"].isin(self.states)]
        future_capacities = future_capacities.rename(
            columns={
                "Planned Year": "Year",
                "State Code": "State",
                "Fuel Source": "Type",
                "Nameplate Capacity": "Capacity",
            }
        )

        future_capacities = future_capacities.drop(
            columns=["Producer Type", "Generators", "Facilities", "Summer Capacity"]
        )

        future_capacities = self.OneHotEncode(future_capacities)
        future_capacities["Capacity"] = future_capacities["Capacity"].str.replace(",", "").astype(float)
        future_capacities = future_capacities.pivot_table(index=["Year", "State"], columns="Type", values="Capacity", fill_value=0).reset_index()

        # making sure that the pred data has the same shape as the data our model was trained on
        missing_columns = set(x_train.columns) - set(future_capacities.columns)

        for column in missing_columns:
            future_capacities[column] = 0

        future_capacities = future_capacities[x_train.columns]

        future_capacities["Total"] = future_capacities.drop(columns=["Year", "State", "Customers"]).sum(axis=1)
        future_capacities = future_capacities[x_train.columns]
        
        if self.modeltype == "linear":
            return self.makeFloat(modelObject.apply_linear_model(future_capacities, modelObject.models))
        else: # Space to add more complex models
            raise ValueError("Invalid model type.")

    def summary_statistics(self, df, name):
        """
        Prints the summary statistics of the training, testing and prediction dataframes.

        Args:
            df (df): The dataframe we want the summary statistics of.
            name (str): The name of the dataframe.
        """
        print(name + " Summary Statistics")
        print(df.describe())

        pass

    def OneHotEncode(self, df, column="State"):
        """
        One hot Encodes the State column of the df.

        Args:
            df (df): The df we want to one hot encode.

        Returns:
            df: The one hot encoded df.
        """
        data = df[column]
        uniq = np.unique(data)
        encoded_states = np.eye(len(uniq))[np.searchsorted(uniq, data)]
        df[column] = [tuple(encoded_states[i]) for i in range(encoded_states.shape[0])]
        return df
    
    def makeFloat(self, df):
        """
        Converts the df to float except the State column.

        Args:
            df (df): The df we want to convert to float.

        Returns:
            df: The converted df.
        """
        columns_to_exclude = ['State']
        dtypes_dict = {col: np.float32 if col not in columns_to_exclude else None for col in df.columns}
        return df.astype(dtypes_dict)
    
    def makeTensor(self, df):
        """
        Converts the df to a tensor, making it compatible with tensorflow.

        Args:
            df (df): The dataframe we want to convert to a tensor.

        Returns:
            tensor: The converted tensor.
        """
        state_array = np.array(df['State'].tolist())
        df_values = df.drop(columns=['State']).values.astype(np.float32)
        return tf.convert_to_tensor(np.concatenate([df_values, state_array], axis=1), dtype=self.tf_data_type)
    
    def makeTensor_Y(self, df):
        """
        Converts the df to a tensor, making it compatible with tensorflow. This function 
        is specifically for the y_train and y_test dataframes, because they don't have the
        State column.

        Args:
            df (df): The dataframe we want to convert to a tensor.

        Returns:
            tensor: The converted tensor.
        """
        return tf.convert_to_tensor(df, dtype=self.tf_data_type)
    
    def minibatch(self):
        """
        Generates a random batch from the training data.

        Returns:
            Tuple: Training batched params.
        """
        batch_id = np.random.choice(self.x_train.shape[0], self.bs, replace=False)
        x_train = tf.gather_nd(self.x_train, tf.expand_dims(batch_id, axis=-1))
        y_train = tf.gather_nd(self.y_train, tf.expand_dims(batch_id, axis=-1))

        return x_train, y_train

    def testbatch(self):
        """
        Generates a random batch from the test data.

        Returns:
            Tuple: Testing batched params.
        """
        batch_id = np.random.choice(self.x_test.shape[0], self.tsbs, replace=False)
        x_test = tf.gather_nd(self.x_test, tf.expand_dims(batch_id, axis=-1))
        y_test = tf.gather_nd(self.y_test, tf.expand_dims(batch_id, axis=-1))

        return x_test, y_test
    
    # TODO: Normalise the data

    # TODO: Reformat data to look at the derivative of the various data, eg population growth instead of absolute population
    
    # TODO: Add data processing for new population data, see consumer_growth.py