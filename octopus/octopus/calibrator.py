from __future__ import print_function

import os.path
import pandas as pd

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError


class Extract:
    def sheet_pull(self):
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
                flow = InstalledAppFlow.from_client_secrets_file(
                    "credentials.json", SCOPES
                )
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open("token.json", "w") as token:
                token.write(creds.to_json())

        try:
            service = build("sheets", "v4", credentials=creds)

            # Call the Sheets API
            sheet = service.spreadsheets()
            result = (
                sheet.values()
                .get(spreadsheetId=SPREADSHEET_ID, range=RANGE_NAME)
                .execute()
            )
            values = result.get("values", [])

            if values:
                return pd.DataFrame(values)
            else:
                print("No data found.")
                return

        except HttpError as err:
            print(err)
