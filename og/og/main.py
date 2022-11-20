import os
import calibrator

if __name__ == "__main__":
    predicted_df, accuracy, cv_score = calibrator.reconnect(os.getenv("CALIBRATOR_LABEL_NAME"))
    print(predicted_df.tail(10))
    print(f"Accuracy: {accuracy}")
    print(f"CV Score: {cv_score}")
