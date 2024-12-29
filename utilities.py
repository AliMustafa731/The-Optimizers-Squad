from sklearn.metrics import roc_curve
import pandas as pd, numpy as np, os
from models import get_compiled_siamese_model

#found from stack overflow
def find_optimal_cutoff(target, predicted):
    """ Find the optimal probability cutoff point for a classification model related to event rate
    Parameters
    ----------
    target : Matrix with dependent or target data, where rows are observations

    predicted : Matrix with predicted data, where rows are observations

    Returns
    -------     
    list type, with optimal cutoff value
        
    """
    fpr, tpr, threshold = roc_curve(target, predicted)
    i = np.arange(len(tpr)) 
    roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
    roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]

    return list(roc_t['threshold'])


siamese_model = get_compiled_siamese_model()
siamese_model.load_weights(os.path.join('Data', 'Models', 'Fine tuned model using LFW', 'model.weights.h5'))
def get_distance_between_faces(face_image1, face_image2):
    prediction = siamese_model.predict(x = [np.expand_dims(face_image1, axis = 0), np.expand_dims(face_image2, axis = 0)])    
    return prediction[0]

