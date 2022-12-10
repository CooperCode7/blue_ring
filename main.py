import os
import og.og.calibrator as calibrator

if __name__ == "__main__":

    LABEL_NAME = os.getenv("CALIBRATOR_LABEL_NAME")

    user_input = input("What data would you like to see? ").lower()

    if user_input == 'prediction table':
        calibrator.get_prediction_table(LABEL_NAME)
    elif user_input == 'feature importance':
        calibrator.get_feature_importance(LABEL_NAME)
    elif user_input == 'decision tree':
        calibrator.get_decision_tree(LABEL_NAME)
    else:
        print("Incorrect argument supplied.")
