import numpy as np


def cv_split(X, Y, cv_i):
    """
    """
    X_valid, y_valid = X[cv_i], Y[cv_i]
    X_train = [x for i, x in enumerate(X) if i != cv_i]
    y_train = [y for i, y in enumerate(Y) if i != cv_i]
    X_train = np.concatenate(X_train)
    y_train = np.concatenate(y_train)

    return X_train, y_train, X_valid, y_valid


def interpolate_A(A_df, pool_df):
    """
    """
    dt0_ids = pool_df.index.get_indexer(A_df['dt0'], method='nearest')
    dt1_ids = pool_df.index.get_indexer(A_df['dt1'], method='nearest')
    pool_df['A'] = np.nan
    pool_df['A'].iloc[dt0_ids] = A_df['A_new']
    pool_df['A'].iloc[dt1_ids] = A_df['A_old']
    ids = pool_df[pool_df['A'].notna()].index
    A_prev = pool_df.loc[ids[0], 'A']
    for id in ids[1:]:
        A_curr = pool_df.loc[id, 'A']
        A = pool_df.loc[id, 'A']
        if A_prev >= A_curr:
            pool_df.loc[id, 'A'] = np.nan
        A_prev = A

    if np.isnan(pool_df.iloc[0]['A']):
        pool_df.iloc[0]['A'] = A_df.iloc[0]['A_old']

    if np.isnan(pool_df.iloc[-1]['A']):
        pool_df.iloc[-1]['A'] = A_df.iloc[-1]['A_old']

    # pool_df['A'].interpolate(method='linear')
    pool_df['A'].interpolate(method='index', inplace=True)

    return pool_df


def sort_results(results):
    """
    """
    f1_list = []
    for res in results:
        f1_list.append(res['f1'])

    f1_list = np.array(f1_list)
    sorted_ids = np.argsort(f1_list)[::-1]

    sorted_results = []
    for id in sorted_ids:
        sorted_results.append(results[id])

    max_f1 = .0
    for res in results:
        if res['f1'] > max_f1:
            max_f1 = res['f1']
            best = res

    return best, sorted_results


def get_D(x_1, y_1, A):
    """
    """
    D_1 = 1 / (3 * 2 ** (1/3))
    D_2 = 432 * A * y_1 * x_1 ** 2 + 432 * A * x_1 * y_1 ** 2
    D_3 = np.sqrt(np.float64(6912 * (4 * A - 1) ** 3 * x_1 ** 3 * y_1 ** 3 +
                  (432 * A * (x_1 ** 2) * y_1 + 432 * A * x_1 * y_1 ** 2) ** 2))

    D_4 = 4 * 2 ** (1/3) * (4 * A - 1) * x_1 * y_1
    D = D_1 * (D_2 + D_3) ** (1/3) - D_4 / ((D_2 + D_3) ** (1/3))

    return D


def get_dy_curve(x, A, D):
    """
    """

    s = D/2
    dy_curve_nom = (1 - 4 * A) * s * x**2 - \
        x * np.sqrt(x * (2 * A * x + s) * (2 * A * (x - 2 * s)**2 + s * x)) + \
        2 * A * x**3 - 2 * s**3

    dy_curve_denom = 2 * x * np.sqrt(x * (2 * A * x + s) *
                                     (2 * A * (x - 2 * s)**2 + s * x))

    return dy_curve_nom / dy_curve_denom


def get_virtualprice(A_contract, x_1, y_1):
    """
    https://www.desmos.com/calculator/mbocz299dv

    """
    A = A_contract / 2

    D = get_D(x_1, y_1, A)
    price = - get_dy_curve(x_1, A, D)

    return price


# def get_y_curve(x, D, A):
#     """
#     x_1 : amount of coin x
#     y_1 : amount of coint y
#     """
#
#     s = D/2
#     y_curve = -x/2 - s/(4*A) + s
#     y_curve += np.sqrt((2*A*x**2 + s*x - 4*A*s*x)**2 + 8*A*x*s**3) / 4*A*x
#
#     return y_curve
