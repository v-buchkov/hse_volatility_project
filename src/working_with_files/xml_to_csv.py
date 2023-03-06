"""Script for extracting xml MOEX fixing data into csv"""
import os
from xml.dom import minidom
from datetime import datetime as dt
from typing import List, Tuple


def read_xml(path: str, directory: str, filename: str, **kwargs) -> List[Tuple[dt, float]]:
    """
    Reads XML, returns List[timestamp, price].

        Parameters:
            path (str): Path to directory (usually, global)
            directory (str): Directory with file
            filename (str): Name of XML file without extension (e.g., input "CNHRUB" for CNHRUB.xml file)

        Returns:
            data_output (list): List of timestamp-price pairs tuples
    """
    directory = directory + '/'
    try:
        file = [f for f in os.listdir(path + directory) if filename in f][0]
    except IndexError:
        raise IndexError(f'No data for {filename} found')
    if 'csv' in file:
        pass
    elif 'xml' in file:
        xml_data = minidom.parse(path + directory + filename + '.xml')
        data_output = [(dt.strptime(item.attributes['TRADEDATE'].value, '%Y-%m-%d'),
                        float(item.attributes['CLOSE'].value)) for item in xml_data.getElementsByTagName('row')]
        data_output.sort(key=lambda x: x[0])
        return data_output
    else:
        raise f'Unknown file format: {file.split(".")[1]}'


def save_to_csv(path: str, directory: str, filename: str, input_data: List[Tuple[dt, float]], **kwargs) -> None:
    """
    Records List[timestamp, price] intp CSV file.

        Parameters:
            path (str): Path to directory (usually, global)
            directory (str): Directory with file
            filename (str): Name of XML file without extension (e.g., input "CNHRUB" for CNHRUB.xml file)
            input_data (list): List of timestamp-price pairs tuples

        Returns:
            None
    """
    directory = directory + '/'
    with open(path + directory + filename + '.csv', 'w') as f:
        for line in input_data:
            date, price = line
            f.write(f'{date},{price}\n')


if __name__ == '__main__':

    directory_path = '../../data'
    price_source = 'moex_fixing'
    assets = ['EURRUB', 'USDRUB', 'EURUSD', 'CNHRUB']

    for a in assets:
        save_to_csv(input_data=read_xml(directory=price_source, filename=a, path=directory_path),
                    directory=price_source, filename=a, path=directory_path)
