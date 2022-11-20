The raw data is collected by a Google Form and stored in a Google Sheet.

### Getting Started
* Create a Google Form in your Gmail account.
 * The form should only consist of questions with true/false answers.
* Columns can be called anything, but a good rule of thumb is to select one column that will be predicted on based on behaviors tracked in the other columns.
 * Example: good_mood is the column to predict and strength_training is one of the behaviors tracked.
* Fill out a response to the Form.
* Create a Google Sheet from the Form. Give the sheet a one word name (ex. Tracker).
* Name the tab with the responses using one word (ex. responses).
* Follow the "Set-up Your Environment" section here to get your necessary credentials and token files: https://developers.google.com/sheets/api/quickstart/python
* Copy the Spreadsheet ID in the URL from the Google Sheet. It follows the /spreadsheets/d/ section of the URL (ex. 1JSDFIF_BR74942JFIJBSLWEO932I0438DJFHEJK)
* Create a local environment variable called GOOGLE_SHEET_ID and assign the Spreadsheet ID to it.
 * Example: ```export GOOGLE_SHEET_ID="1JSDFIF_BR74942JFIJBSLWEO932I0438DJFHEJK"```
* Create another local environment variable called GOOGLE_SHEET_RANGE and assign the tab name with the response data to it.
 * Example: ```export GOOGLE_SHEET_RANGE="responses"```
* Create another local environment variable called CALIBRATOR_LABEL_NAME and assign the column you want to predict on.
 * Example: ```export CALIBRATOR_LABEL_NAME="good_mood"``` 
* TBD - waiting on the app to be finished before providing further instructions.