from __future__ import print_function

import os.path
import pandas as pd
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import matplotlib.pyplot as pyplot

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

    # The ID and range of the Google Sheet.
    SPREADSHEET_ID = os.getenv("GOOGLE_SHEET_ID")
    RANGE_NAME = os.getenv("GOOGLE_SHEET_RANGE")

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

    # Reset the index to address the re-sort
    response_df = response_df.reset_index(drop=True)

    # Keep an original version before dropping the Timestamp so it can be
    # connected to the predictions. Convert the string TRUE/FALSE valeus to
    # boolean while leaving missing and datetime data intact
    timestamp_df = response_df
    timestamp_df = timestamp_df.applymap(lambda x: True if x == "TRUE" else x)
    timestamp_df = timestamp_df.applymap(lambda x: False if x == "FALSE" else x)

    # Drop the Timestamp since it's no longer necessary
    response_df = response_df.drop("Timestamp", axis=1)

    # Convert the remaining TRUE/FALSE values to 1/0
    # Missing values should remain empty since those will be handled by the
    # Iterative Imputer
    response_df = response_df.applymap(lambda x: 1 if x == "TRUE" else x)
    response_df = response_df.applymap(lambda x: 0 if x == "FALSE" else x)

    # Have to mask empty columns that aren't handled by the above code so they
    # are treated as NaN for the Imputer
    response_df = response_df.mask(response_df == "")

    return response_df, timestamp_df


def days_before(label_name):
    """This function will add true to a set number of periods (aka days) before the
    first true label value for a given time block. This helps catch creeping events
    that are not consciously known, but are present."""

    adjusted_df, _ = prep_data()

    # Get the actual days to offset by subtracting the var from 0.
    days_offset = 0 - int(os.getenv("DAYS_OFFSET", 0))

    # Add the label diff field used to determine how many days to set backwards
    # from the start of the event
    adjusted_df["label_diff"] = (
        adjusted_df[label_name].diff(periods=days_offset).fillna(0)
    )

    # Update the label value based on the results of the label_diff value
    # Have to do a diff < 0 since the future true (1) - the current row false (0) = -1
    adjusted_df[label_name] = adjusted_df.apply(
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
    feat_df = pd.DataFrame(data=imp_mean.transform(prep_df)).astype(int)

    # Assign the actual column names to the imperative imputer dataframe
    feat_df.columns = prep_df.columns
    feat_df.index = prep_df.index

    return feat_df, lab_df


def train_test(label_name):
    """Train the data to be used in the Random Forest algorithm."""

    # Bring in the necessary DataFrames and assign them to X/y variables
    feat_df, lab_df = tree_data(label_name)
    X = feat_df
    y = lab_df

    # Get the training and testing values
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=1
    )

    return X_train, X_test, y_train, y_test


def classify(label_name):
    """Take the trained data and use the Random Forest algorithm to classify
    each record"""

    # Import the feature DataFrame to be used for the final prediction
    feat_df, _ = tree_data(label_name)

    # Unpack the values in the tuple from the previous functions
    X_train, X_test, y_train, y_test = train_test(label_name)

    # Create the classifier model with the default parameters.
    classifier = RandomForestClassifier()

    # Create the parameters to be used in cross validation.
    # Starting small for now since max depth and max_features had the most impact
    param_dict = {"max_depth": [3, 5, 7, 9], "max_features": [0.20, 0.33, "sqrt"]}

    # Apply the cross validation function to the existing classifier
    grid_clf = GridSearchCV(estimator=classifier, param_grid=param_dict)

    # Fit the dataset to the new random forest classifier with cross validation
    model = grid_clf.fit(X_train, y_train)

    # Run the test model for accuracy scores
    test_pred = model.predict(X_test)

    # Run the model against the entire DataFrame and return a list of predictions
    final_pred = list(model.predict(feat_df))

    # Provide simple accuracy scores.
    accuracy = metrics.accuracy_score(y_test, test_pred)
    cv_score = model.best_score_

    return final_pred, accuracy, cv_score, model, feat_df


def reconnect(label_name):
    """Take the list of predictions and add them as a column to the original dataframe."""

    # Pull in the necessary objects
    _, timestamp_df = prep_data()
    final_pred, accuracy, cv_score, _, _ = classify(label_name)

    # Assign the final prediction list to a column on the dataframe
    timestamp_df["predicted_event"] = final_pred

    # Convert the new column values to boolean
    timestamp_df["predicted_event"] = timestamp_df["predicted_event"].astype(bool)

    return timestamp_df, accuracy, cv_score


def get_prediction_table(label_name):
    """Output a table with the last 90 days' worth of predictions."""

    # Pull in the necessary objects
    predicted_df, accuracy, cv_score = reconnect(label_name)

    # Set the DataFrame to show all rows
    pd.set_option("display.max_rows", None)

    # # Print out the prediction vs. actual for the last 90 days as well as the accuracy scores
    print(predicted_df[["Timestamp", "event", "predicted_event"]].tail(90))
    print("Accuracy: " + "{:.0%}".format(accuracy))
    print(f"CV Score: " + "{:.0%}".format(cv_score))


def get_feature_importance(label_name):
    """Create a feature importance visualization to show which features have the
    greatest impact."""

    _, _, _, model, feat_df = classify(label_name)
    feat_importances = pd.Series(
        model.best_estimator_.feature_importances_, index=feat_df.columns
    )
    feat_importances.sort_values(ascending=True, inplace=True)
    feat_importances.plot(kind="barh")
    pyplot.title("Feature Importances")
    pyplot.show()
