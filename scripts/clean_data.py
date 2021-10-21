def clean_data(x):
    # clean the data
    mean_x = np.mean(x)
    row_id, col_id = np.where(x == -999.0)
    x[row_id, col_id] = np.mean(x)
    return x
