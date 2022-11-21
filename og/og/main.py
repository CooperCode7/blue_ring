import os
import pandas as pd
import calibrator

# Wrap in this if statement to prevent accidental runs on import
if __name__ == "__main__":

    label_name = os.getenv("CALIBRATOR_LABEL_NAME")

    # Pull in the necessary objects
    predicted_df, accuracy, cv_score = calibrator.reconnect(label_name)

    # # Print out the prediction vs. actual for the last 30 days as well as the accuracy scores
    print(predicted_df[['Timestamp','event', 'predicted_event']].tail(30))
    print(f"Accuracy: {accuracy}")
    print(f"CV Score: {cv_score}")

    # Create the feature importance visualization
    calibrator.prioritize(label_name)
