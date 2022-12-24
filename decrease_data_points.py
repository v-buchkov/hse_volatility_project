import pandas as pd

if __name__ == '__main__':
    PATH = 'data/'
    SOURCE = 'moex'
    ASSET = 'USDRUB'

    data_points = pd.read_csv(PATH + SOURCE + '/' + ASSET + '.csv')
    data_points['timestamp'] = pd.to_datetime(data_points['timestamp']).dt.strftime('%Y-%m-%d %H')

    data_points = data_points.drop_duplicates(subset=['timestamp'])

    data_points['timestamp'] = pd.to_datetime(data_points['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')

    data_points.to_csv(PATH + SOURCE + '/' + ASSET + '.csv', index=False)

    print(data_points)
