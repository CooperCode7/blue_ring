from __future__ import print_function

import os.path
import json
import pandas as pd
from sqlalchemy import create_engine

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

    # The ID and range of a sample spreadsheet.
    SPREADSHEET_ID = "1so7AoYxZ2NVG2IHdU4u8pQ2cZfEAGyj4GpqcIeR0hYg"
    RANGE_NAME = "responses"

    # Create the credentials.json file for Heroku
    if os.getenv("DEV_LOCATION", "local") == "heroku":
        JSON_SECRET = json.dumps(os.getenv("GOOGLE_CLIENT_SECRETS"))
        with open("credentials.json", "w") as f:
            f.write(JSON_SECRET)

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
            print(pd.DataFrame(values))
            return pd.DataFrame(values)
        else:
            print("No data found.")
            return

    except HttpError as err:
        print(err)


def pg_conn():
    """Establish the connection to the postgres database."""

    # Heroku stores the dialect as "postgres" but SQLAlchemy requires "postgresql"
    conn_string = os.getenv("DATABASE_URL")
    if conn_string and conn_string.startswith("postgres://"):
        conn_string = conn_string.replace("postgres://", "postgresql://", 1)

    db = create_engine(conn_string)
    return db.connect()


def data_push():
    """Push the data from the dataframe to the postgres database."""

    with pg_conn() as conn:
        sql_df = sheet_pull()
        sql_df.to_sql("tracker", con=conn, if_exists="replace", index=False)
