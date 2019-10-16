# -*- coding: utf-8 -*-
"""
Created on Sat Jun  1 18:17:23 2019

@author: miche
"""


class Temp:
    # This class is built for temporary files
    def __init__(self,ID):
        self.ID = ID
        self.pct = 0
        self.filename = []
        self.data = []
        
    def update(self):
        import os
        import pickle
        for pct in range(99,0,-1): # assuming the files are save every 1% at max precision
            if os.path.isfile(self.ID+'_'+str(int(pct))+'%.p'):
                self.filename = self.ID+'_'+str(int(pct))+'%.p'
                self.pct = int(pct)
                self.data = pickle.load(open(self.filename,"rb"))
        return True
    
    def save(self,data,pct):
        import os
        import pickle
        
        old_filename = self.filename
        self.filename = self.ID + '_'+str(int(pct))+'%.p'
        self.pct = int(pct)
        self.data = data
        pickle.dump(self.data,open(self.filename,"wb"))
        if old_filename:
            try:
                os.remove(old_filename)
            except:
                pass
        return True
    
    def delete(self):
        import os
        if not self.filename:
            raise ValueError('The temp file was never saved and therefore cannot be deleted')
        try:
            os.remove(self.filename)
        except:
            raise ValueError('Could not delete the temp file because it does not exist')
        return True
        
        
        

# Test for the Temp class:
if __name__ == '__main__':
    import os, pickle, random, sys
    from io import StringIO
    os.chdir('C:\\Users\\miche\\OneDrive\\Charite\\Projects\\Backpropagation\\BP_TF2\\tmp')
    ID = 'tempID'
    temp_object = Temp(ID)
    for i in range(0,30,5):
        data = [random.random()]*10
        temp_object.save(data,i)
        if os.path.isfile(temp_object.filename):
            print('The file was saved succesfully!')
    recent_filename = temp_object.filename
    temp_object = Temp(ID)
    temp_object.update()
    if temp_object.filename==recent_filename:
        print('The update was done successfully!')
    temp_object.delete()
    if not(os.path.isfile(temp_object.filename)):
        print('The temp file was deleted successfully!')
    
        
        


def last_to_linear(model):
    ''' The function gets a keras sequential model and returns the model with the
    last layer with linear activation instead of softmax '''
    #imports
    import os
    from keras import activations
    from keras.models import Sequential, Model,load_model

    model.layers[-1].activation = activations.linear # for some reason it changes the activation function in "model" as well as "BP_model" 

    # Code taken from utils.apply_modifications instead of running "BP_model = utils.apply_modifications(BP_model)" due to fucking bugs
    tmp_model_name = 'tmp_model.h5'
    model.save(tmp_model_name)
    model = load_model(tmp_model_name)
    os.remove(tmp_model_name)
    return model


def wavg_saliency_grads(model,class_idx,inputs,input_weights,last_layer_idx=-1,backprop_modifier='guided',ID = 'temp'):
    ''' The function calculates the weighted average saliency_map gradient based on
    the inputs (images) and the corresponding input_scores
    input variables:
        model = keras sequential model with the last layer switched from softmax/sigmoid to linear
        class_idx = idx of the predicted class (out of M columns of input_scores). 
            please notice that the class index is limitted to the number of output channels of the last layer. This is being checked within the function and will raise ValueError if it's not the case.
        inputs = 3D array of size [N,k,q] of the input images or [N,k] for tabular data,
            for which the grads should be calculated, where N is the number of inputs (observations) 
            and [k x q] is the image size or k = no. of input features for the tabular data case.
        input_scores = 2D array of size [N,M] which represents the weights corresponding to each set of the inputs variable and class,
            different weighting can amplify the weight of different inputs and/or different classes. 
            A typical weighting of the inputs could be yielded by relation of the prediction-scores to the ground-truth class
        last_layer_idx = the index of the last layer of model
        backprop_modifier = is used further in the visualize_saliency function
    output:
        wavg_grad = [k,q] array of the weighted average saliency gradient for class class_idx
    '''
    # Import relevant modules:    
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import keras,os
    from keras.models import Sequential, Model,load_model
    from keras.layers import Dense, Dropout, Flatten, Activation, Input
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    from vis.visualization import visualize_cam_with_losses, visualize_cam, visualize_activation, visualize_saliency
    from vis.utils import utils
    from keras import activations
    
    if len(inputs.shape)==3: # 2D input features i.e. images
        # Run tests:
        [N,k,q] = inputs.shape
        [NN,M] = input_weights.shape
        if NN != N:
            raise ValueError('The no. of observations in inputs = ',N,' does not match to input_weights = ',NN)          
        wavg_grad = np.zeros([k,q]) # initiate the weighted grads as zero
        
    elif len(inputs.shape)==2: # 1D input features i.e. tabular-data
        # Run tests:
        [N,k] = inputs.shape
        [NN,M] = input_weights.shape
        if NN != N:
            raise ValueError('The no. of observations in inputs = ',N,' does not match to input_scores = ',NN)
        wavg_grad = np.zeros([k]) # initiate the weighted grads as zero
        
    else: # different dimensionality of input data is not supported (yet) but could be easily extended if needed
        raise ValueError('The wavg function currently does not support input dimensions of',inputs.shape[1:])
    
    # Make sure that the class-idx corresponds to the range of possible class idx (i.e. within the range of model's last layer no. of channels)
    if class_idx<0 or class_idx > model.get_layer(index = last_layer_idx).output_shape[1]-1:
        raise ValueError('The specified class_idx = ',class_idx,' does not correspond to the number of output channels = ',model.get_layer(index = last_layer_idx).output_shape[1])
    
    # Create a temporary object and see if there are temporary results saved for the given ID
    temp_results = Temp(ID)
    temp_results.update()
    
    if temp_results.filename: # if temporary results are already saved, upload them and continue from the last recorded percentage, 
        wavg_grad = temp_results.data
        start = int(temp_results.pct*N/100)
    else: # otherwise start from scratch:
        start = 0
    #start = 0
    print('the weighted average saliency is being calculated..')
    for i in range(start,N):
        if type(inputs)==np.ndarray: # for inputs of type ndarray 
            wavg_grad += input_weights[i][class_idx]*visualize_saliency(model, last_layer_idx, filter_indices=class_idx, seed_input=inputs[i],backprop_modifier=backprop_modifier,keepdims=True)
        elif type(inputs)==pd.core.frame.DataFrame: # for inputs of type DataFrame
            wavg_grad += input_weights[i][class_idx]*visualize_saliency(model, last_layer_idx, filter_indices=class_idx, seed_input=inputs.iloc[i],backprop_modifier=backprop_modifier,keepdims=True)
        else:
            raise TypeError('The inputs data type = ',type(inputs),' is not supported')
        
        if i%(N/100)==0: # print calculation progression
            print(100*i/N,'% done!')
            # save temporary file with temporary results:
            temp_results.save(wavg_grad,100*i/N)

    temp_results.delete()        
    return wavg_grad


def wavg_cam_grads(model,class_idx,inputs,input_weights,last_layer_idx=-1,backprop_modifier='guided',ID = 'temp'):
    ''' The function calculates the weighted average saliency_map gradient based on
    the inputs (images) and the corresponding input_scores
    input variables:
        model = keras sequential model with the last layer switched from softmax/sigmoid to linear
        class_idx = idx of the predicted class (out of M columns of input_scores). 
            please notice that the class index is limitted to the number of output channels of the last layer. This is being checked within the function and will raise ValueError if it's not the case.
        inputs = 3D array of size [N,k,q] of the input images or [N,k] for tabular data,
            for which the grads should be calculated, where N is the number of inputs (observations) 
            and [k x q] is the image size or k = no. of input features for the tabular data case.
        input_scores = 2D array of size [N,M] which represents the weights corresponding to each set of the inputs variable and class,
            different weighting can amplify the weight of different inputs and/or different classes. 
            A typical weighting of the inputs could be yielded by relation of the prediction-scores to the ground-truth class
        last_layer_idx = the index of the last layer of model
        backprop_modifier = is used further in the visualize_saliency function
    output:
        wavg_grad = [k,q] array of the weighted average saliency gradient for class class_idx
    '''
    # Import relevant modules:    
    import numpy as np
    import pandas as pd
    import tensorflow as tf
    import matplotlib.pyplot as plt
    import keras,os
    from keras.models import Sequential, Model,load_model
    from keras.layers import Dense, Dropout, Flatten, Activation, Input
    from keras.layers import Conv2D, MaxPooling2D
    from keras import backend as K
    from vis.visualization import visualize_cam_with_losses, visualize_cam, visualize_activation, visualize_saliency
    from vis.utils import utils
    from keras import activations
    
    if len(inputs.shape)==3: # 2D input features i.e. images
        # Run tests:
        [N,k,q] = inputs.shape
        [NN,M] = input_weights.shape
        if NN != N:
            raise ValueError('The no. of observations in inputs = ',N,' does not match to input_weights = ',NN)          
        wavg_grad = np.zeros([k,q]) # initiate the weighted grads as zero
        
    elif len(inputs.shape)==2: # 1D input features i.e. tabular-data
        # Run tests:
        [N,k] = inputs.shape
        [NN,M] = input_weights.shape
        if NN != N:
            raise ValueError('The no. of observations in inputs = ',N,' does not match to input_scores = ',NN)
        wavg_grad = np.zeros([k]) # initiate the weighted grads as zero
        
    else: # different dimensionality of input data is not supported (yet) but could be easily extended if needed
        raise ValueError('The wavg function currently does not support input dimensions of',inputs.shape[1:])
    
    # Make sure that the class-idx corresponds to the range of possible class idx (i.e. within the range of model's last layer no. of channels)
    if class_idx<0 or class_idx > model.get_layer(index = last_layer_idx).output_shape[1]-1:
        raise ValueError('The specified class_idx = ',class_idx,' does not correspond to the number of output channels = ',model.get_layer(index = last_layer_idx).output_shape[1])
    
    # Create a temporary object and see if there are temporary results saved for the given ID
    temp_results = Temp(ID)
    temp_results.update()
    
    if temp_results.filename: # if temporary results are already saved, upload them and continue from the last recorded percentage, 
        wavg_grad = temp_results.data
        start = int(temp_results.pct*N/100)
    else: # otherwise start from scratch:
        start = 0
    #start = 0
    print('the weighted average saliency is being calculated..')
    for i in range(start,N):
        if type(inputs)==np.ndarray: # for inputs of type ndarray 
            wavg_grad += input_weights[i][class_idx]*visualize_saliency(model, last_layer_idx, filter_indices=class_idx, seed_input=inputs[i],backprop_modifier=backprop_modifier,keepdims=True)
        elif type(inputs)==pd.core.frame.DataFrame: # for inputs of type DataFrame
            wavg_grad += input_weights[i][class_idx]*visualize_saliency(model, last_layer_idx, filter_indices=class_idx, seed_input=inputs.iloc[i],backprop_modifier=backprop_modifier,keepdims=True)
        else:
            raise TypeError('The inputs data type = ',type(inputs),' is not supported')
        
        if i%(N/100)==0: # print calculation progression
            print(100*i/N,'% done!')
            # save temporary file with temporary results:
            temp_results.save(wavg_grad,100*i/N)

    temp_results.delete()        
    return wavg_grad


    