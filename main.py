import os
import og.og.calibrator as calibrator

if __name__ == "__main__":

    label_name = os.getenv("CALIBRATOR_LABEL_NAME")

    user_input = input("What data would you like to see? ").lower()

    if user_input == 'prediction table':
        calibrator.get_prediction_table(label_name)
    elif user_input == 'feature importance':
        calibrator.get_feature_importance(label_name)
    else:
        print("Incorrect argument supplied.")
