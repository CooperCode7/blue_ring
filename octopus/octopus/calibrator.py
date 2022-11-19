from __future__ import print_function

import os.path
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics


from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


def sheet_pull():
    """Pulls data from the Google Sheet into a Pandas dataframe. This will
    subsequently be loaded into a Postgres database."""

    # If modifying these scopes, delete the file token.json.
    SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

    # The ID and range of a sample spreadsheet.
    SPREADSHEET_ID = "1so7AoYxZ2NVG2IHdU4u8pQ2cZfEAGyj4GpqcIeR0hYg"
    RANGE_NAME = "responses"

    creds = None

    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists("token.json"):
        creds = Credentials.from_authorized_user_file("token.json", SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file("credentials.json", SCOPES)
            creds = flow.run_local_server(port=0)

        # Save the credentials for the next run
        with open("token.json", "w") as token:
            token.write(creds.to_json())

    try:
        service = build("sheets", "v4", credentials=creds)

        # Call the Sheets API
        sheet = service.spreadsheets()
        result = (
            sheet.values().get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME).execute()
        )
        values = result.get("values", [])

        if values:
            df = pd.DataFrame(values)

            # Fix the dataframe so the first row is treated as the header row
            df.columns = df.iloc[0]
            df = df[1:]
            return df
        else:
            print("No data found.")
            return

    except HttpError as err:
        print(err)


def prep_data():
    """Cleans the dataframe to make subsequent transformations smoother."""

    # Read the Google Sheet dataframe
    response_df = sheet_pull()

    # Identify the date columns.
    datetime_columns = ["Timestamp"]

    # Convert the datetime_columns list
    response_df[datetime_columns] = response_df[datetime_columns].apply(pd.to_datetime)

    # Sort the dataframe so the latest response is always last
    response_df = response_df.sort_values(by=["Timestamp"])

    # Reset the index since older values from an older response sheet were apeneded
    response_df = response_df.reset_index(drop=True)

    # drop the Timestamp since it's no longer necessary
    response_df = response_df.drop("Timestamp", axis=1)

    # Convert the remaining TRUE/FALSE values to 1/0
    # Missing values should remain empty since those will be handled by the
    # Iterative Imputer
    response_df = response_df.applymap(lambda x: 1 if x == "TRUE" else x)
    response_df = response_df.applymap(lambda x: 0 if x == "FALSE" else x)

    # Have to mask empty columns that aren't handled by the above code so they
    # are treated as NaN for the Imputer
    response_df = response_df.mask(response_df == "")

    return response_df


def days_before(label_name):
    """This function will add true to a set number of periods (aka days) before the
    first true label value for a given time block. This helps catch creeping events
    that are not consciously known, but are present."""

    adjusted_df = prep_data()

    # For the below, have to convert the true/false label to int because the
    # underlying numpy operation on diff will throw a deprecation warning.
    adjusted_df["label_diff"] = adjusted_df[label_name].diff(periods=-5).fillna(0)
    adjusted_df[label_name] = adjusted_df.apply(
        # Have to do a diff <0 since the future true (1) - the current row false (0) = -1
        lambda x: 1 if x[label_name] == 0 and x["label_diff"] < 0 else x[label_name],
        axis=1,
    ).astype(int)

    # Drop the label_diff column since it should not be included in subsequent predictions
    adjusted_df = adjusted_df.drop("label_diff", axis=1)

    return adjusted_df


def tree_data(label_name):
    """This function will be used to create a dataframe that can be used for testing
    and predicting with decision tree and random forest algorithms"""

    # Create the base dataframe to use for the algorithms
    base_df = days_before(label_name)

    # This sequence will pull in the dataframes necessary to train and test the model
    prep_df = base_df.drop(label_name, axis=1)
    lab_df = base_df[label_name]

    # Drop any column that has all nan (not a number) values. If a column has all
    # nan, the imputer won't work
    prep_df = prep_df.dropna(how="all", axis=1)

    # Use iterative imputer to fill in missing values. The median stretegy is necessary
    # since the missing values can date back months or even years. Median is better suited
    # for skewed distributions that could result from that.
    imp_mean = IterativeImputer(initial_strategy="median", max_iter=100)
    imp_mean.fit(prep_df)

    # Create a new features data frame with the imputed value
    feat_df = pd.DataFrame(data=imp_mean.transform(prep_df), dtype="int")

    # Assign the actual column names to the imperative imputer dataframe
    feat_df.columns = prep_df.columns
    feat_df.index = prep_df.index

    return feat_df, lab_df, prep_df


# Practice Runs
# df = prep_data()
# print(df)
# print(days_before("event"))
# feat_df, lab_df = tree_data("event")
# print(feat_df)
