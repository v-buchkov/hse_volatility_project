from __future__ import print_function

import io
import os.path
from typing import Dict, Union, Any

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']


def authorize_and_get_creds() -> Credentials:
    creds = None

    if os.path.exists('token.json'):
        creds = Credentials.from_authorized_user_file('token.json', SCOPES)

    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file('credentials.json', SCOPES)
            creds = flow.run_local_server(port=0)

        with open('token.json', 'w') as token:
            token.write(creds.to_json())

    return creds


def get_data_id(path: str) -> Union[Dict, str]:
    if path[-1] == '/':
        path = path.split('/')[0]

    creds = authorize_and_get_creds()

    try:
        service = build('drive', 'v3', credentials=creds)

        folder_id = (service.files().list(q=f"name='{path.split('/')[0]}'").execute()).get('files', [])[0]['id']

        results = service.files().list(q=f"'{folder_id}' in parents").execute()
        items = results.get('files', [])

        if not items:
            raise FileNotFoundError('No files found.')

        file_tree = create_file_tree(root_folder_id=folder_id, creds=creds)

        file_id_step = file_tree.copy()
        for p in path.split('/')[1:]:
            if p in file_id_step.keys():
                file_id_step = file_id_step[p]
            else:
                raise FileNotFoundError('No files found.')

        return file_id_step

    except HttpError as error:
        print(f'An error occurred: {error}')
        return


def get_file_by_path(path: str) -> Any:
    return get_file_by_id(get_data_id(path))


def get_file_by_id(file_id: str):
    creds = authorize_and_get_creds()

    try:
        service = build('drive', 'v3', credentials=creds)

        file_request = service.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, file_request)

        done = False
        while done is False:
            status, done = downloader.next_chunk()
        fh.seek(0)

        return fh.read()

    except HttpError as error:
        print(f'An error occurred: {error}')
        return


def create_file_tree(root_folder_id: str, creds: Credentials) -> Dict:
    # TODO: use class to create a tree
    file_tree_dict = {}

    service = build('drive', 'v3', credentials=creds)

    results = service.files().list(q=f"'{root_folder_id}' in parents").execute()
    items = results.get('files', [])

    for item in items:
        if item['mimeType'] == 'application/vnd.google-apps.folder':
            file_tree_dict[item['name']] = create_file_tree(item['id'], creds=creds)
        else:
            file_tree_dict[item['name']] = item['id']

    return file_tree_dict


if __name__ == '__main__':
    print()
