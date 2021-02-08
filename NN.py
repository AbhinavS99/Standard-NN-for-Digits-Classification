import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.manifold import TSNE

class MyNeuralNetwork():
    """
    My implementation of a Neural Network Classifier.
    """

    acti_fns = ['relu', 'sigmoid', 'linear', 'tanh', 'softmax']
    weight_inits = ['zero', 'random', 'normal']

    def __init__(self, n_layers, layer_sizes, activation, learning_rate, weight_init, batch_size, num_epochs):
        """
        Initializing a new MyNeuralNetwork object

        Parameters
        ----------
        n_layers : int value specifying the number of layers

        layer_sizes : integer array of size n_layers specifying the number of nodes in each layer

        activation : string specifying the activation function to be used
                     possible inputs: relu, sigmoid, linear, tanh

        learning_rate : float value specifying the learning rate to be used

        weight_init : string specifying the weight initialization function to be used
                      possible inputs: zero, random, normal

        batch_size : int value specifying the batch size to be used

        num_epochs : int value specifying the number of epochs to be used
        """

        if activation not in self.acti_fns:
            raise Exception('Incorrect Activation Function')

        if weight_init not in self.weight_inits:
            raise Exception('Incorrect Weight Initialization Function')
        
        
        self.n_layers = n_layers
        self.layer_sizes = layer_sizes
        self.activation = activation
        self.learning_rate = learning_rate
        self.weight_init = weight_init
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.test_switch = False
        self.tsne_switch = False
        
        if self.activation == "relu":
            self.act_func = self.relu
            self.act_func_grad = self.relu_grad
            
        elif self.activation == "sigmoid":
            self.act_func = self.sigmoid
            self.act_func_grad = self.sigmoid_grad
            
        elif self.activation == "tanh":
            self.act_func = self.tanh
            self.act_func_grad = self.tanh_grad
        
        elif self.activation == "linear":
            self.act_func = self.linear
            self.act_func_grad = self.linear_grad
        
        self.weights = self.transform([])
        self.biases = self.transform([])
        
        for i in range(0,self.n_layers-1):
            self.biases.append(np.zeros((self.layer_sizes[i+1],1)))
            if self.weight_init == "zero":
                w = self.zero_init((self.layer_sizes[i+1],self.layer_sizes[i]))
            elif self.weight_init == "random":
                w = self.random_init((self.layer_sizes[i+1],self.layer_sizes[i]))
            elif self.weight_init == "normal":
                w = self.normal_init((self.layer_sizes[i+1],self.layer_sizes[i]))     
            self.weights.append(w)
            
    
    def transform(self, obj):
        obj.insert(0,0)
        return obj
    
    def relu(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.maximum(X,0)

    def relu_grad(self, X):
        """
        Calculating the gradient of ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        x_calc = np.zeros(X.shape)
        x_calc[X<=0] = 0
        x_calc[X>0] = 1
        return x_calc

    def sigmoid(self, X):
        """
        Calculating the Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1/(1 +np.exp(-X))

    def sigmoid_grad(self, X):
        """
        Calculating the gradient of Sigmoid activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return self.sigmoid(X)*(1-self.sigmoid(X))

    def linear(self, X):
        """
        Calculating the Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return X

    def linear_grad(self, X):
        """
        Calculating the gradient of Linear activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.ones(X.shape)

    def tanh(self, X):
        """
        Calculating the Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return np.tanh(X)

    def tanh_grad(self, X):
        """
        Calculating the gradient of Tanh activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return 1 - np.square(np.tanh(X))

    def softmax(self, X):
        """
        Calculating the ReLU activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        exp = np.exp(X - np.max(X))
        return exp / exp.sum(axis = 0, keepdims = True)

    def softmax_grad(self, X):
        """
        Calculating the gradient of Softmax activation for a particular layer

        Parameters
        ----------
        X : 1-dimentional numpy array 

        Returns
        -------
        x_calc : 1-dimensional numpy array after calculating the necessary function over X
        """
        return None

    def zero_init(self, shape):
        """
        Calculating the initial weights after Zero Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.zeros(shape)

    def random_init(self, shape):
        """
        Calculating the initial weights after Random Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.rand(shape)*0.01

    def normal_init(self, shape):
        """
        Calculating the initial weights after Normal(0,1) Activation for a particular layer

        Parameters
        ----------
        shape : tuple specifying the shape of the layer for which weights have to be generated 

        Returns
        -------
        weight : 2-dimensional numpy array which contains the initial weights for the requested layer
        """
        return np.random.normal(0,1,shape)*0.01
    
    
    def generate_batches(self, x, y):
        if len(x) != len(y):
            raise Exception('Input data does not have equal number of features and labels')
        
        num_batches = len(x)//self.batch_size
        op = []
        for i in range(0,num_batches):
            temp = (i*self.batch_size,(i+1)*self.batch_size)
            fin = [x for x in range(temp[0],temp[1])]
            op.append(fin)
        return op
    
    
    def feed_forward(self, inp):
        a = inp.T
        
        activations = self.transform([])
        pre_activations = self.transform([])
        weights = self.transform([])
        
        for i in range(0,len(self.weights)-2):
            z = self.weights[i+1].dot(a) + self.biases[i+1]
            a = self.act_func(z)
            
            weights.append(self.weights[i+1])
            activations.append(a)
            pre_activations.append(z)
            
        ind = len(self.weights) - 1
        z = self.weights[ind].dot(a) + self.biases[ind]
        a = self.softmax(z)
        
        weights.append(self.weights[ind])
        activations.append(a)
        pre_activations.append(z)
        
        return a, activations, pre_activations, weights
    
    def cross_entropy_loss(self, y_true, y_pred):
        noise = 1e-8
        return -1*np.mean(y_true*np.log(y_pred.T + noise))
    
    
    def back_propagate(self, x, y, activations, pre_activations, weights):
        dW, dB = [0]*len(self.weights), [0]*len(self.biases)
        activations[0] = x.T
        
        A = activations[len(self.weights)-1]
        dZ = A - y.T
        dAPrev = weights[len(self.weights)-1].T.dot(dZ)
        
        dw = dZ.dot(activations[len(self.weights)-2].T)/x.shape[0]
        db = np.sum(dZ, axis = 1, keepdims = True)/x.shape[0]
        
        dW[len(self.weights)-1] = dw
        dB[len(self.weights)-1] = db
        
        for i in range(len(self.weights)-2, 0, -1):
            dZ = dAPrev*self.act_func_grad(pre_activations[i])
            
            dw = dZ.dot(activations[i-1].T)/x.shape[0]
            db = np.sum(dZ, axis = 1, keepdims = True)/x.shape[0]
            
            if i > 1:
                dAPrev = self.weights[i].T.dot(dZ)
                
            dB[i] = db
            dW[i] = dw
            
        return dW, dB
    
    def update_params(self, dW, dB):
        for i in range(1, len(self.weights)):
            self.weights[i] = self.weights[i] - self.learning_rate*dW[i]
            self.biases[i] = self.biases[i] - self.learning_rate*dB[i]
            
    def set_test_switch(self, val):
        self.test_switch = val

    def set_tsne_switch(self, val):
        self.tsne_switch = val
    
    def assign_test_sets(self, x_test, y_test):
        self.x_test = x_test
        self.y_test = y_test
        enc = OneHotEncoder(sparse = False, categories ='auto')
        self.y_test_enc = enc.fit_transform(y_test.reshape(len(y_test),-1))

    def calc_test_loss(self):
        if self.test_switch:
            a, _, _ , _ = self.feed_forward(self.x_test)
            test_cost = self.cross_entropy_loss(self.y_test_enc, a)
            self.test_costs.append(test_cost)
            

    def fit(self, X, y):
        """
        Fitting (training) the linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as training data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as training labels.
        
        Returns
        -------
        self : an instance of self
        """
        self.costs = []
        self.test_costs = []
        enc = OneHotEncoder(sparse = False, categories ='auto')
        Y = enc.fit_transform(y.reshape(len(y),-1))
        
        for epoch in range(0,self.num_epochs):
            batch_cost = 0
            for idx in self.generate_batches(X,y):
                x_batch, y_batch = X[idx], Y[idx]
                
                y_pred, activations, pre_activations, weights = self.feed_forward(x_batch)
                batch_cost += self.cross_entropy_loss(y_batch, y_pred)
                
                dW, dB = self.back_propagate(x_batch, y_batch, activations, pre_activations, weights)
                self.update_params(dW,dB)
                
            num_batches = len(X)/self.batch_size
            costs = batch_cost/num_batches
            self.costs.append(costs)
            self.calc_test_loss()
            
            if epoch%10 == 0:
                print("\nFor epoch ==> "+str(epoch+1))
                print("Loss ==> "+str(costs))
                
        print("\nFor epoch ==> "+str(self.num_epochs))
        print("Loss ==> "+str(self.costs[-1]))
        
        return self

    def predict_proba(self, X):
        """
        Predicting probabilities using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 2-dimensional numpy array of shape (n_samples, n_classes) which contains the 
            class wise prediction probabilities.
        """

        activation, _, _, _ = self.feed_forward(X)
        return activation

    def predict(self, X):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        Returns
        -------
        y : 1-dimensional numpy array of shape (n_samples,) which contains the predicted values.
        """

        y_pred = self.predict_proba(X)
        y_pred = np.argmax(y_pred, axis = 0)
        return y_pred

    def score(self, X, y):
        """
        Predicting values using the trained linear model.

        Parameters
        ----------
        X : 2-dimensional numpy array of shape (n_samples, n_features) which acts as testing data.

        y : 1-dimensional numpy array of shape (n_samples,) which acts as testing labels.

        Returns
        -------
        acc : float value specifying the accuracy of the model on the provided testing set
        """

        y_pred = self.predict(X)
        acc = 0
        for i in range(0,len(y)):
            if y_pred[i] == y[i]:
                acc = acc + 1
        return acc/len(X)
    
    
    def plot_loss_graphs(self):
        plt.figure()
        title_str = "For Activation Function ==> "+self.activation
        plt.plot(range(self.num_epochs), self.costs, 'r', label = 'Training Loss')
        plt.plot(range(self.num_epochs), self.test_costs, 'b', label = 'Testing Loss')
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title(title_str)
        plt.legend()
        plt.show()
    
    def plot_tsne(self, layer_num):
        if self.tsne_switch:
            tsne = TSNE(n_components = 2)
            a, activations, _, _ = self.feed_forward(self.x_test)
            a = activations[layer_num]
            a = a.T
            a = tsne.fit_transform(a)
            y_test = self.y_test
            y_test = y_test.reshape(len(y_test),1)
            data = np.append(a,y_test,axis=1)
            data = pd.DataFrame(data=data,columns=['PC1','PC2','y'])
            plt.figure(figsize=(12,8))
            sns.scatterplot(x ="PC1", y = "PC2", hue = "y", palette = sns.color_palette("hls",10),data= data,legend = "full")
            plt.show()
    
    def save_model(self, filename):
        filename = './models/'+filename+'.pkl'
        joblib.dump(self, filename)
        print("Model saved......")