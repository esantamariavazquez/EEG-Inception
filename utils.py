from sklearn.preprocessing import OneHotEncoder

def categorical_labels(one_hot_labels):
    cat_labels = np.argmax(one_hot_labels, axis=1)
    return cat_labels

def one_hot_labels(caategorical_labels):
    enc = OneHotEncoder(handle_unknown='ignore')
    on_hot_labels = enc.fit_transform(caategorical_labels.reshape(-1, 1)).toarray()
    
    return on_hot_labels