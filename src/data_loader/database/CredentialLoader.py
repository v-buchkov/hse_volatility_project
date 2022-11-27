import json
import os


class CredentialLoader:
    CREDENTIALS_FILENAME = "credentials.json"
    CREDENTIALS_FOLDER = 'CREDENTIALS_FOLDER'

    @classmethod
    def load_credentials(cls, name, what="storage", clip_first_host=True, change_last_to_star=False):
        cur_dir_files = [f for f in os.listdir(os.getcwd()) if os.path.isfile(os.path.join(os.getcwd(), f))]

        # check for credentials file in: environment variable, working directory or home folder
        if cls.CREDENTIALS_FOLDER in os.environ:
            folder_path = os.environ['CREDENTIALS_FOLDER']
        elif cls.CREDENTIALS_FILENAME in cur_dir_files:
            folder_path = os.getcwd()
        else:
            folder_path = os.environ['HOME']

        path = os.path.join(folder_path, cls.CREDENTIALS_FILENAME)
        with open(path) as f:
            big_dict = json.load(f)
            return Credentials(big_dict, what, name, clip_first_host, change_last_to_star)


class Credentials:

    def __init__(self, big_dict, what, name, clip_first_host, change_last_to_star):
        target_dict = big_dict[what][name]
        self.name = name
        self.cred_type = what
        if what == "storage":

            self.host = target_dict['host']
            if clip_first_host and ',' in self.host:
                self.host = self.host[:self.host.find(',')]
                if change_last_to_star:
                    # strip away higher-level domain if present,
                    # i.e. server1.domain.com -> server*
                    if '.' in self.host:
                        self.host = self.host[:self.host.find('.') - 1] + '*'
                    else:
                        self.host = self.host[:-1] + '*'
            self.port = target_dict['port']
            self.login = target_dict['login']
            self.password = target_dict['password']
            try:
                self.database = target_dict['database']
            except KeyError:
                self.database = None
        else:
            raise ValueError("Unknown credentials type")
