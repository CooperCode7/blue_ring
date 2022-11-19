# blue_ring
Welcome! This app represents the rebuilding of a two year old pet project to track daily activities and make a prediction on whether the day was positive or negative. This will help users determine if they need to make modifications to their daily activities. The raw data is collected by a Google Form and stored in a Google Sheet. There is an item on the roadmap to develop a custom survey app using Flask to acquire this data.

Currently, the majority of the working code is located in the "OG" folder: https://github.com/NuclearOctopus/blue_ring/tree/main/og/og. This code is reminiscient of the original project. However, it includes improvements to the efficiency of the code and reduces the reliance on 3rd-party libraries.

### Getting Started
* Create a Google Form in your Gmail account.
 * The form should only consist of questions with true/false answers.
 * Columns can be called anything, but a good rule of thumb is to select one column that will be predicted on based on behaviors tracked in the other columns. Example: daily_mood is the column to predict and strength_training is one of the behaviors tracked.
* Fill out a response to the Form.
* Create a Google Sheet from the Form. Give the sheet a one word name (ex. Tracker).
* Name the tab with the responses. using one word (ex. responses).
* Copy the Spreadsheet ID in the URL from the Google Sheet. It follows the /spreadsheets/d/ section of the URL (ex. 1JSDFI_FBR74942JFIJBSLWEO932I0438DJFHEJK)
* Create a local environment variable called GOOGLE_SHEET_ID and assign the Spreadsheet ID to it.
 * Example: ```export GOOGLE_SHEET_ID="1JSDFI_FBR74942JFIJBSLWEO932I0438DJFHEJK"```
* Create another local environment variable called GOOGLE_SHEET_RANGE and assign the tab name with the response data to it.
 * Example: ```export GOOGLE_SHEET_RANGE="responses"```
* TBD