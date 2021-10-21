def pca_decomposition(data, num_components):
    '''
    Linear dimensionality reduction using Singular Value Decomposition of the data
    to project it to a lower dimensional space
    '''
    data_meaned = data - np.mean(data)
    # calculating the covariance matrix of the mean-centered data
    covariance_m = np.cov(data_meaned , rowvar = False)
    
    # calculating eigenvalues and eigenvectors of the covariance matrix
    e_vals , e_vecs = np.linalg.eigh(covariance_m)
    
    # sort the eigenvalues in descending order
    sorted_id = np.argsort(e_vals)[::-1]
    e_vals_sorted = e_vals[sorted_id]
    
    # sort the eigenvectors 
    e_vecs_sorted = e_vecs[:,sorted_id]
    
    # select the first num_components eigenvectors
    selected_e_vecs = e_vecs_sorted[:,:num_components]
    
    # transform the data 
    pca_components = np.dot(selected_e_vecs.transpose(),data_meaned.transpose()).transpose()
    
    return pca_components
