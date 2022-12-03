import os
import pandas as pd
import calibrator


def get_prediction_table(label_name):
    """Output a table with the last 90 days' worth of predictions."""

    # Pull in the necessary objects
    predicted_df, accuracy, cv_score = calibrator.reconnect(label_name)

    # Set the DataFrame to show all rows
    pd.set_option("display.max_rows", None)

    # # Print out the prediction vs. actual for the last 90 days as well as the accuracy scores
    print(predicted_df[["Timestamp", "event", "predicted_event"]].tail(90))
    print("Accuracy: " + "{:.0%}".format(accuracy))
    print(f"CV Score: " + "{:.0%}".format(cv_score))


def get_feature_importance(label_name):
    """Create a visualization of feature importances tied to the predictions."""

    # Create the feature importance visualization
    calibrator.prioritize(label_name)


if __name__ == "__main__":

    label_name = os.getenv("CALIBRATOR_LABEL_NAME")

    user_input = input("What data would you like to see? ").lower()

    if user_input == 'prediction table':
        get_prediction_table(label_name)
    elif user_input == 'feature importance':
        get_feature_importance(label_name)
    else:
        print("Incorrect argument supplied.")
