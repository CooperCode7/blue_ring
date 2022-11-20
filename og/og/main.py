import os
import pandas as pd
import calibrator

# Wrap in this if statement to prevent accidental runs on import
if __name__ == "__main__":

    # Want to see all the columns in the dataframe
    pd.set_option("display.max_columns", None)

    # Pull in the necessary objects
    predicted_df, accuracy, cv_score = calibrator.reconnect(
        os.getenv("CALIBRATOR_LABEL_NAME")
    )

    # Print out the dataframe as well as the scores
    print(predicted_df.tail(10))
    print(f"Accuracy: {accuracy}")
    print(f"CV Score: {cv_score}")
