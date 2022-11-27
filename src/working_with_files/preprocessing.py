import numpy as np
from datetime import datetime as dt
from typing import List, Dict, Tuple
import os
from xml.dom import minidom
import openpyxl

XLSX_SKIP = 9


def reader_decorator(func):
    """
    Decorates function that preprocesses some markets data.

        Parameters:
            func (function): Processing function to decorate (e.g., getting returns of some asset).

        Returns:
            file_reader (function): Decorating function.
    """
    def file_reader(path: str, directory: str, filename: str, **kwargs):
        """
        Reads given file that contains data about the asset.

            Parameters:
                path (str): Path to directory (usually, global)
                directory (str): Directory with file
                filename (str): Name of XML file without extension (e.g., input "CNHRUB" for CNHRUB.xml file)
                **kwargs : Other keyword arguments that will be passed into the function that is being decorated

            Returns:
                called_function (function): Calls function that is being decorated.
        """
        directory = directory + '/'

        try:
            # Get all files from the directory with the specified filename
            files_in_directory = [f for f in os.listdir(path + directory) if filename in f]
            # Checks that the filename is unique. If not, raises the AssertionError
            assert len(set(files_in_directory)) == len(files_in_directory), f'Ambiguous file {filename} for ' \
                                                                            f'directory {directory}'
            # Take the applicable file. If no files were found, IndexError will be called
            file = files_in_directory[0]
        except IndexError:
            raise IndexError(f'No {directory.rstrip("/").upper()} data for {filename} found')

        # Check file extension, and call applicable function
        extension = file.split('.')[1]
        if extension == 'csv':
            input_list = csv_reader(path + directory, filename)
        elif extension == 'xml':
            input_list = xml_reader(path + directory, filename)
        elif extension == 'xlsx':
            input_list = xlsx_reader(path + directory, filename)
        else:
            # If extension is different from available, raise Error
            raise f'Unknown file format: {file.split(".")[1]}'
        return func(filename=filename, input_data=input_list, **kwargs)
    return file_reader


def csv_reader(path_to_file: str, filename: str) -> List[Tuple]:
    """
    Reads CSV, returns List[timestamp, price].

        Parameters:
            path_to_file (str): Path to file (global path + directory merged)
            filename (str): Name of CSV file without extension (e.g., input "CNHRUB" for CNHRUB.csv file)

        Returns:
            data_output (list): List of timestamp-price pairs tuples
    """
    with open(path_to_file + filename + '.csv') as csv_file:
        csv_data = [line for line in csv_file][1:]
        return [tuple([dt.strptime(line.split(',')[0].split('+')[0].split('.')[0], '%Y-%m-%d %H:%M:%S'),
                       float(line.split(',')[1])]) for line in csv_data if line.split(',')[1] != '']


def xml_reader(path_to_file: str, filename: str) -> List[Tuple[dt, float]]:
    """
    Reads XML, returns List[timestamp, price].

        Parameters:
            path_to_file (str): Path to file (global path + directory merged)
            filename (str): Name of XML file without extension (e.g., input "CNHRUB" for CNHRUB.xml file)

        Returns:
            data_output (list): List of timestamp-price pairs tuples
    """
    xml_data = minidom.parse(path_to_file + filename + '.xml')
    xml_data = [(dt.strptime(item.attributes['TRADEDATE'].value, '%Y-%m-%d'),
                 float(item.attributes['CLOSE'].value)) for item in xml_data.getElementsByTagName('row')]
    xml_data.sort(key=lambda x: x[0])
    return xml_data


def xlsx_reader(path_to_file: str, filename: str) -> List[Tuple[dt, float]]:
    """
    Reads XLSX, returns List[timestamp, price].

        Parameters:
            path_to_file (str): Path to file (global path + directory merged)
            filename (str): Name of XLSX file without extension (e.g., input "CNHRUB" for CNHRUB.xlsx file)

        Returns:
            data_output (list): List of timestamp-price pairs tuples
    """
    xlsx_data = [line for line in openpyxl.open(path_to_file + filename + '.xlsx').active.values][XLSX_SKIP:]
    xlsx_data.sort(key=lambda x: x[0])
    return xlsx_data


def opt_data_filename_decorator(func):
    """
    Decorates function with specified asset and date for option data into MOEX FORTS file naming format.

        Parameters:
            func (function): Processing function to decorate (e.g., getting returns of some asset).

        Returns:
            file_reader (function): Decorating function.
    """
    def transformer_asset_name(**kwargs) -> str:
        """
        Transforms given asset name and date for option data into filename.

            Parameters:
                **kwargs : Keyword arguments that will be passed into the function that is being decorated

            Returns:
                called_function (function): Calls function that is being decorated.
        """
        month = int(kwargs["month"])
        if month < 10:
            month = '0' + str(month)
        return func(directory=f'{kwargs["price_source"]}', filename=f'{kwargs["year"]}{month}_opt_deal', **kwargs)
    return transformer_asset_name


def return_data_filename_decorator(func):
    """
    Decorates function with specified asset and pricing source into usual file naming format.

        Parameters:
            func (function): Processing function to decorate (e.g., getting returns of some asset).

        Returns:
            file_reader (function): Decorating function.
    """
    def transformer_asset_name(**kwargs) -> str:
        """
        Transforms given asset name and pricing source into filename.

            Parameters:
                **kwargs : Keyword arguments that will be passed into the function that is being decorated

            Returns:
                called_function (function): Calls function that is being decorated.
        """
        return func(directory=f'{kwargs["price_source"]}', filename=f'{kwargs["asset"]}', **kwargs)
    return transformer_asset_name


def transform_option_deals_into_dict(func):
    """
    Decorates function with specified options input data into list of dicts with deal data.

        Parameters:
            func (function): Processing function to decorate (e.g., getting returns of some asset).

        Returns:
            file_reader (function): Decorating function.
    """
    def list_of_dicts(**kwargs):
        """
        Transforms given input data into list of dicts with deal data.

            Parameters:
                **kwargs : Keyword arguments that will be passed into the function that is being decorated

            Returns:
                called_function (function): Calls function that is being decorated.
        """
        input_data = kwargs['input_data']
        # Keys is the first line of the file => first list in the input_data
        keys = input_data[0]
        return func(deals=[{keys[i]: deal[i] for i in range(len(keys))} for deal in input_data[1:]], **kwargs)
    return list_of_dicts


@return_data_filename_decorator
@reader_decorator
def get_asset_returns(input_data: List[Tuple[dt, float]], delta_seconds: int, **kwargs) -> Dict[dt, float]:
    """
    Generates list of asset returns.
    Applies reader decorator to get and read appropriate file by specified asset name.
    Applies filename decorator to change asset name into filename.

        Parameters:
            input_data (list): Processing function to decorate (e.g., getting returns of some asset).
            delta_seconds (int): Timestamp difference between returns in seconds (e.g., 8 * 60 * 60 for daily spacing).

        Returns:
            output_dict (dict): Asset's {timestamp: return} dict.
    """
    output_dict = {}

    # Calculate appropriate index spacing for the list of returns to match specified delta_seconds
    for j in range(0, len(input_data)):
        time_delta = abs((input_data[j][0] - input_data[0][0]).total_seconds())
        if time_delta >= delta_seconds:
            break

    # Generates list of returns with adjustment (getting return by specified unit of time)
    # Iteration with step of j, found earlier as appropriate index spacing parameter
    for i in range(0, len(input_data) - j, j):
        # Calculate, how many seconds is contained in one step of iteration by index
        # Adjust by specified change in time
        adj_coefficient = abs((input_data[i + j][0] - input_data[i][0]).total_seconds()) / time_delta
        # Generate returns in the form of log-price difference (better features of distribution than r[i+1] / r[i] - 1)
        output_dict[input_data[i + j][0]] = (np.log(input_data[i + j][1]) - np.log(input_data[i][1])) / adj_coefficient

    return output_dict


@return_data_filename_decorator
@reader_decorator
def get_asset_prices(input_data: List[Tuple[dt, float]], delta_seconds: int, **kwargs) -> Dict[dt, float]:
    """
    Generates list of asset prices.
    Applies reader decorator to get and read appropriate file by specified asset name.
    Applies filename decorator to change asset name into filename.

        Parameters:
            input_data (list): Processing function to decorate (e.g., getting returns of some asset).
            delta_seconds (int): Timestamp difference between returns in seconds (e.g., 9 * 60 * 60 for daily spacing).

        Returns:
            output_dict (dict): Asset's {timestamp: price} dict.
    """
    output_dict = {}

    # Calculate appropriate index spacing for the list of returns to match specified delta_seconds
    for j in range(0, len(input_data)):
        time_delta = abs((input_data[j][0] - input_data[0][0]).total_seconds())
        if time_delta >= delta_seconds:
            break

    for i in range(0, len(input_data) - j, j):
        # Generate prices list, passing some prices inside delta_seconds interval (includes only significant spaces)
        output_dict[input_data[i][0]] = input_data[i][1]

    return output_dict


@opt_data_filename_decorator
@reader_decorator
@transform_option_deals_into_dict
def get_option_prices(**kwargs) -> List[dict]:
    """
    Generates list of MOEX option deals {deal_parameter: value} dictionaries.
    Applies reader decorator to get and read appropriate file by specified asset name.
    Applies filename decorator to change asset name into filename.
    Applies list of lists into list of dicts decorator.

        Parameters:
            **kwargs : Deal keyword arguments

        Returns:
            asset_deals (dict): Asset's list of {deal_parameter: value} dicts.
    """
    asset = kwargs['asset']
    # Get asset code on MOEX (e.g., for USDRUB use "Si")
    asset_code_moex = kwargs['code_moex']
    deals = kwargs['deals']
    asset_deals = []
    # Iterate over deals
    for deal in deals:
        # If the deal was with needed asset
        if asset_code_moex in str(deal['#SYMBOL']):
            # Deal side (buy or sell)
            side = deal['DIRECTION\n'].rstrip('\n')
            # Transform into datetime
            deal_date = dt(year=int(deal['MOMENT'][:4]), month=int(deal['MOMENT'][4:6]), day=int(deal['MOMENT'][6:8]))
            # Call or put
            option_type = deal['SYSTEM']
            # Strike - contained into MOEX option name
            strike = float(deal['#SYMBOL'].lstrip(asset_code_moex)[:5]) / 1000
            # Option price in RUB
            price = float(deal['PRICE_DEAL'])
            # Volume - # of options on the deal
            volume = int(deal['VOLUME'])
            asset_deals.append({'date': deal_date, 'asset': asset, 'side': side, 'option_type': option_type,
                                'strike': strike, 'price': price, 'volume': volume})
    return asset_deals
