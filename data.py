import pandas as pd
import numpy as np
from utils import get_virtualprice, interpolate_A
from duneanalytics import DuneAnalytics
import json


def write_A_dict(A_dict, fname):
    """
    Writes to a json file.
    """
    with open(fname, 'w') as f:
        json.dump(A_dict, f)


def read_A_dict(fname):
    """
    Reads a json file of A parameter values writtent by write_A_dict()
    """
    with open(fname, 'r') as f:
        A_df = pd.DataFrame(json.load(f))

    A_df['dt0'] = pd.to_datetime(A_df['ts0'], unit='s', utc=True)
    A_df['dt1'] = pd.to_datetime(A_df['ts1'], unit='s', utc=True)

    return A_df


def prepare_price(win=5, threshold=1., fill='ffill'):
    """
    Argumments
    ----------
    win
    threshold - absolute, in percent
    fill - Fill NaN with previous value ("ffil") or subsequent value ("bfill").
           Both methods give the same result.

    """
    prices = []
    pool_names = []

    tmp = get_pool(pool_fname='data/pool/pool_curve_usdn_USDN-3CRV.csv',
                   A_fname='data/A/A_curve_usdn_USDN-3CRV.json',
                   tok0='USDN', tok1='3CRV')
    prices.append(tmp['price'].resample('24h').mean().fillna(method=fill))
    pool_names.append('USDN-3CRV')

    tmp = get_pool(pool_fname='data/pool/pool_curve_degenbox_MIM-UST.csv',
                   A_fname='data/A/A_curve_degenbox_MIM-UST.json',
                   tok0='ust', tok1='mim')
    prices.append(tmp['price'].resample('24h').mean().fillna(method=fill))
    pool_names.append('MIM-UST')

    # tmp = get_pool(pool_fname='data/pool/pool_curve_degenbox_MIM-UST.csv',
    #                A_fname='data/A/A_curve_degenbox_MIM-UST.json',
    #                tok0='ust', tok1='mim')

    tmp = get_pool(pool_fname='data/pool/pool_curve_sETH-ETH.csv',
                   A_fname='data/A/A_curve_sETH-ETH.json',
                   tok0='steth_pool', tok1='eth_pool')
    prices.append(tmp['price'].resample('24h').mean().fillna(method=fill))
    pool_names.append('sETH-ETH')

    tmp = get_pool(pool_fname='data/pool/pool_curve_pUSd_pUSd-3Crv.csv',
                   A_fname='data/A/A_curve_pUSd_pUSd-3Crv.json',
                   tok0='pusd', tok1='threecrv')
    prices.append(tmp['price'].resample('24h').mean().fillna(method=fill))
    pool_names.append('pUSd-3Crv')

    tmp = get_pool(pool_fname='data/pool/pool_curve_wormhole-ust_UST-3Pool.csv',
                   A_fname='data/A/A_curve_wormhole-ust_UST-3Pool.json',
                   tok0='ust', tok1='three_pool')
    prices.append(tmp['price'].resample('24h').mean().fillna(method=fill))
    pool_names.append('UST-3Pool')
    # ts6 = get_3pool(fname='Curve-3Pool_balance_DAI-USDC-USDT.csv')

    threshold = 1 + threshold / 100
    X, Y = [], []
    for price in prices:
        x, y = [], []
        N = price.shape[0] - win - 1
        for i in range(N):
            x.append(price[i:i+win].values)
            y.append(price[i+win])

        x, y = np.array(x), np.array(y)

        y = threshold < (np.abs(x[:, -1] - y) + x[:, -1]) / x[:, -1]

        X.append(x)
        Y.append(y)

    return X, Y, pool_names


def get_pool(pool_fname='data/pool/pool_curve_usdn_USDN-3CRV.csv',
             A_fname='data/A/A_curve_usdn_USDN-3CRV.json',
             tok0='USDN', tok1='3CRV'):
    """

    USDN
    ---------
    pool_fname = 'data/pool/pool_curve_usdn_USDN-3CRV.csv'
    A_fname = 'data/A/A_curve_usdn_USDN-3CRV.json'
    tok0='USDN'
    tok1='3CRV'

    pUSD
    --------
    pool_fname = 'data/pool/pool_curve_pUSd_pUSd-3Crv.csv'
    A_fname = 'data/A/A_curve_pUSd_pUSd-3Crv.json'
    tok0 = 'pusd'
    tok1 = 'threecrv'

    Wormhole UST
    ------------
    pool_fname = 'data/pool/pool_curve_wormhole-ust_UST-3Pool.csv'
    A_fname = 'data/A/A_curve_wormhole-ust_UST-3Pool.json'
    tok0 = 'ust'
    tok1 = 'three_pool'

    Degenbox
    --------
    pool_fname = 'data/pool/pool_curve_degenbox_MIM-UST.csv'
    A_fname = 'data/A/A_curve_degenbox_MIM-UST.json'
    tok0='ust'
    tok1='mim'

    sETH
    ------------------------
    pool_fname='data/pool/pool_curve_sETH-ETH.csv'
    A_fname='data/A/A_curve_sETH-ETH.json'
    tok0='steth_pool'
    tok1='eth_pool'
    """
    # Amplification parameter
    A_df = read_A_dict(A_fname)

    pool_df = pd.read_csv(pool_fname)
    try:
        pool_df.drop('Unnamed: 0', axis=1, inplace=True)
    except KeyError:
        pass

    pool_df['time'] = pd.to_datetime(pool_df['time'])
    pool_df.set_index('time', inplace=True)

    pool_df = interpolate_A(A_df, pool_df)

    pool_df['ratio'] = pool_df[tok0] / pool_df[tok1]

    pool_df['price'] = np.nan
    for i in pool_df.index:
        x_1 = np.float64(pool_df.loc[i, tok0])
        y_1 = np.float64(pool_df.loc[i, tok1])
        A_contract = np.float64(pool_df.loc[i, 'A'])
        price = get_virtualprice(A_contract, x_1, y_1)
        pool_df.loc[i, 'price'] = price

    return pool_df


def get_3pool(pool_fname='data/pool/pool_curve_3Pool_DAI-USDC-USDT.csv',
              A_fname='data/A/A_curve_3Pool_DAI-USDC-USDT.json'):
    """
    """
    # Amplification parameter
    A_df = read_A_dict(A_fname)

    df = pd.read_csv(pool_fname)
    df['time'] = pd.to_datetime(df['UNIX TimeStamp'], unit='s', utc=True)
    df = df.sort_values(by='time', ascending=True)
    df.set_index('time', inplace=True)
    df['DAI Balance'] = df['DAI Balance'].astype('float64')/1e18
    df[' USDT Balance'] = df[' USDT Balance'].astype('float64')/1e6
    df['USDC Balance'] = df['USDC Balance'].astype('float64')/1e6
    df['price'] = df['Virtual Price'].astype('float64')/1e18

    df = interpolate_A(A_df, df)

    df['ratio_USDT/USDC'] = df[' USDT Balance'] / df['USDC Balance']
    df['ratio_DAI/USDC'] = df['DAI Balance'] / df['USDC Balance']
    df['ratio_DAI/USDT'] = df['DAI Balance'] / df[' USDT Balance']

    df['price_USDT/USDC'] = np.nan
    df['price_DAI/USDC'] = np.nan
    df['price_DAI/USDT'] = np.nan
    for i in df.index:
        x_1 = np.float64(df.loc[i, ' USDT Balance'])
        y_1 = np.float64(df.loc[i, 'USDC Balance'])
        A_contract = np.float64(df.loc[i, 'A'])
        price = get_virtualprice(A_contract, x_1, y_1)
        df.loc[i, 'price_USDT/USDC'] = price

        x_1 = np.float64(df.loc[i, 'DAI Balance'])
        y_1 = np.float64(df.loc[i, 'USDC Balance'])
        A_contract = np.float64(df.loc[i, 'A'])
        price = get_virtualprice(A_contract, x_1, y_1)
        df.loc[i, 'price_DAI/USDC'] = price

        x_1 = np.float64(df.loc[i, 'DAI Balance'])
        y_1 = np.float64(df.loc[i, ' USDT Balance'])
        A_contract = np.float64(df.loc[i, 'A'])
        price = get_virtualprice(A_contract, x_1, y_1)
        df.loc[i, 'price_DAI/USDT'] = price

    return df


def download_usdn(username, password, query_id=977508):
    """
    USDN Metapool
    Curve Balance (USDN/DAI/USDT/USDC)
    Query:
        https://dune.com/queries/977508
    Contract:
        https://etherscan.io/address/0x0f9cb53Ebe405d49A0bbdBD291A65Ff571bC83e1
    3CRV:
        https://etherscan.io/address/0x6c3F90f043a72FA612cbac8115EE7e52BDe6E490
    """
    dune = DuneAnalytics(username, password)
    dune.login()
    # fetch token
    dune.fetch_auth_token()
    result_id = dune.query_result_id(query_id)
    # fetch query result
    data = dune.query_result(result_id)

    cols = data['data']['query_results'][0]['columns']

    df = pd.DataFrame(columns=cols)
    for i in range(len(data['data']['get_result_by_result_id'])):
        tmp = pd.DataFrame(data['data']['get_result_by_result_id'][i]['data'],
                           columns=cols, index=[i])
        df = pd.concat([df, tmp])

    df = df.sort_values(by='time', ascending=True)
    df.set_index('time', inplace=True)
    # time to datettime
    df.to_csv('data/pool_curve_usdn_USDN-3CRV.csv')

    return df


def download_wormhole_ust(username, password, query_id=611757):
    """
    Wormhole UST(Curve)
    Curve Balance (UST/3Pool(DAI/USDC/USDT))
    Query:
        https://dune.com/queries/611757
    Contract:
        https://etherscan.io/address/0xCEAF7747579696A2F0bb206a14210e3c9e6fB269
    """
    dune = DuneAnalytics(username, password)
    dune.login()
    # fetch token
    dune.fetch_auth_token()
    result_id = dune.query_result_id(query_id)
    # fetch query result
    data = dune.query_result(result_id)

    cols = data['data']['query_results'][0]['columns']

    df = pd.DataFrame(columns=cols)
    for i in range(len(data['data']['get_result_by_result_id'])):
        tmp = pd.DataFrame(data['data']['get_result_by_result_id'][i]['data'],
                           columns=cols, index=[i])
        df = pd.concat([df, tmp])

    df = df.sort_values(by='time', ascending=True)
    df.set_index('time', inplace=True)
    # time to datettime
    df.to_csv('data/pool_curve_wormhole-ust_UST-3Pool.csv')

    return df


def download_pUSd(username, password, query_id=1032572):
    """
    pUSd-3Crv (Curve)
    Curve Balance (pUSd/3Crv(DAI/USDC/USDT))
    Query:
        https://dune.com/queries/1032572
    Contract:
        https://etherscan.io/address/0x8EE017541375F6Bcd802ba119bdDC94dad6911A1
    """
    dune = DuneAnalytics(username, password)
    dune.login()
    # fetch token
    dune.fetch_auth_token()
    result_id = dune.query_result_id(query_id)
    # fetch query result
    data = dune.query_result(result_id)

    cols = data['data']['query_results'][0]['columns']

    df = pd.DataFrame(columns=cols)
    for i in range(len(data['data']['get_result_by_result_id'])):
        tmp = pd.DataFrame(data['data']['get_result_by_result_id'][i]['data'],
                           columns=cols, index=[i])
        df = pd.concat([df, tmp])

    df = df.sort_values(by='time', ascending=True)
    df.set_index('time', inplace=True)
    # time to datettime
    df.to_csv('data/pool_curve_pUSd_pUSd-3Crv.csv')

    return df


def download_degenbox(username, password, query_id=977257):
    """
    Query:
        https://dune.com/queries/977257
    Contract:
        https://etherscan.io/token/0x55A8a39bc9694714E2874c1ce77aa1E599461E18
    """
    dune = DuneAnalytics(username, password)
    dune.login()
    # fetch token
    dune.fetch_auth_token()
    result_id = dune.query_result_id(query_id)
    # fetch query result
    data = dune.query_result(result_id)

    cols = data['data']['query_results'][0]['columns']

    df = pd.DataFrame(columns=cols)
    for i in range(len(data['data']['get_result_by_result_id'])):
        tmp = pd.DataFrame(data['data']['get_result_by_result_id'][i]['data'],
                           columns=cols, index=[i])
        df = pd.concat([df, tmp])

    df = df.sort_values(by='time', ascending=True)
    df.set_index('time', inplace=True)
    # time to datettime
    df.to_csv('data/pool_curve_degenbox_MIM-UST.csv')

    return df
