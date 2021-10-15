from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class FullyConnectedNet(object):
    """Class for a multi-layer fully connected neural network.

    Network contains an arbitrary number of hidden layers, ReLU nonlinearities,
    and a softmax loss function. This will also implement dropout and batch/layer
    normalization as options. For a network with L layers, the architecture will be

    {affine - [batch/layer norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch/layer normalization and dropout are optional and the {...} block is
    repeated L - 1 times.

    Learnable parameters are stored in the self.params dictionary and will be learned
    using the Solver class.
    """

    def __init__(
        self,
        hidden_dims,
        input_dim=3 * 32 * 32,
        num_classes=10,
        dropout_keep_ratio=1,
        normalization=None,
        reg=0.0,
        weight_scale=1e-2,
        dtype=np.float32,
        seed=None,
    ):
        """Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout_keep_ratio: Scalar between 0 and 1 giving dropout strength.
            If dropout_keep_ratio=1 then the network should not use dropout at all.
        - normalization: What type of normalization the network should use. Valid values
            are "batchnorm", "layernorm", or None for no normalization (the default).
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
            initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
            this datatype. float32 is faster but less accurate, so you should use
            float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers.
            This will make the dropout layers deteriminstic so we can gradient check the model.
        """
        self.normalization = normalization
        self.use_dropout = dropout_keep_ratio != 1
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution centered at 0 with standard       #
        # deviation equal to weight_scale. Biases should be initialized to zero.   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to ones and shift     #
        # parameters should be initialized to zeros.                               #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        # create an array of input dims, hidden dims and number of classes
        all_dims = np.concatenate((input_dim,hidden_dims,num_classes),axis=None)

        for i in range(1,len(all_dims)):
          
          self.params["W{0}".format(i)] = weight_scale*np.random.randn(all_dims[i-1],all_dims[i])
          self.params["b{0}".format(i)] = np.zeros(all_dims[i],)
         
        # Batch Normalization 

        if  self.normalization == 'batchnorm' or self.normalization == 'layernorm': 

          for j in range(1,len(all_dims)-1):
            self.params["gamma{0}".format(j)] = np.ones(hidden_dims[j-1],)
            self.params["beta{0}".format(j)] = np.zeros(hidden_dims[j-1],)


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {"mode": "train", "p": dropout_keep_ratio}
            if seed is not None:
                self.dropout_param["seed"] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.normalization == "batchnorm":
            self.bn_params = [{"mode": "train"} for i in range(self.num_layers - 1)]
        if self.normalization == "layernorm":
            self.bn_params = [{} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype.
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)

    def loss(self, X, y=None):
        """Compute loss and gradient for the fully connected net.
        
        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
            scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
            names to gradients of the loss with respect to those parameters.
        """
        X = X.astype(self.dtype)
        mode = "test" if y is None else "train"

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param["mode"] = mode
        if self.normalization == "batchnorm":
            for bn_param in self.bn_params:
                bn_param["mode"] = mode
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        
        out_hist = {}
        out_hist['out0'] = X
        cache_hist = {}


        # 1. when no normalization is applied, and no dropout, vanilla network.
        if self.normalization == None:
          for i in range(1,self.num_layers):
            out_hist['out{0}'.format(i)], cache_hist['cache{0}'.format(i)] \
                              = affine_relu_forward(out_hist['out{0}'.format(i-1)],
                                self.params['W{0}'.format(i)],
                                self.params['b{0}'.format(i)]) 


          del out_hist['out0']
          i +=1 
          scores, cache_hist['cache{0}'.format(i)]= affine_forward   \
                                  (out_hist['out{0}'.format(i-1)], \
                                  self.params['W{0}'.format(i)],   \
                                  self.params['b{0}'.format(i)])
        
        # 2. when no normalization is applied, but dropout is applied
        elif self.normalization == None and self.use_dropout:
          for i in range(1,self.num_layers):
            out_hist['out{0}'.format(i)], cache_hist['cache{0}'.format(i)] \
                              = affine_relu_dropout_forward(out_hist['out{0}'.format(i-1)],
                                self.params['W{0}'.format(i)],
                                self.params['b{0}'.format(i)],
                                self.dropout_param) 

          del out_hist['out0']
          i +=1 
          scores, cache_hist['cache{0}'.format(i)]= affine_forward   \
                                  (out_hist['out{0}'.format(i-1)], \
                                  self.params['W{0}'.format(i)],   \
                                  self.params['b{0}'.format(i)])
        
        # 3. when batch normalization is applied
        elif self.normalization == 'batchnorm':
          for i in range(1,self.num_layers):     
            out_hist['out{0}'.format(i)], cache_hist['cache{0}'.format(i)] \
                              = affine_batchnorm_relu_forward(out_hist['out{0}'.format(i-1)],
                                self.params['W{0}'.format(i)],
                                self.params['b{0}'.format(i)],
                                self.params['gamma{0}'.format(i)],
                                self.params['beta{0}'.format(i)],
                                self.bn_params[i-1])

          del out_hist['out0']

          i +=1 
          scores, cache_hist['cache{0}'.format(i)]= affine_forward   \
                                  (out_hist['out{0}'.format(i-1)], \
                                  self.params['W{0}'.format(i)],   \
                                  self.params['b{0}'.format(i)])

        # 4. when layers normalization is applied
        elif self.normalization == 'layernorm':

          for i in range(1,self.num_layers):     
            out_hist['out{0}'.format(i)], cache_hist['cache{0}'.format(i)] \
                              = affine_layernorm_relu_forward(out_hist['out{0}'.format(i-1)],
                                self.params['W{0}'.format(i)],
                                self.params['b{0}'.format(i)],
                                self.params['gamma{0}'.format(i)],
                                self.params['beta{0}'.format(i)],
                                self.bn_params[i-1])

          del out_hist['out0']

          i +=1 
          scores, cache_hist['cache{0}'.format(i)]= affine_forward   \
                                  (out_hist['out{0}'.format(i-1)], \
                                  self.params['W{0}'.format(i)],   \
                                  self.params['b{0}'.format(i)])


        '''
        #Only forward pass and relu

        out1, cache1 = affine_relu_forward(X,self.params['W1'],self.params['b1'])
        out1, cache1 = dropout_forward(out1,self.dropout[aramers])
        out2, cache2 = affine_relu_forward(out1,self.params['W2'],self.params['b2'])
        scores, cache3 = affine_forward(out2,self.params['W3'],self.params['b3'])
        
        elif self.normalization == 'batchnorm':

          # Forward pass, batch normalization and relu
          out1, cache1 = affine_batchnorm_relu_forward(X,W1,b1,gamma1,beta1,self.bn_params[0])
          out2, cache2 = affine_batchnorm_relu_forward(out1,W2,b2,gamma2,beta2,self.bn_params[1])
          
        '''
      

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early.
        if mode == "test":
            return scores

        loss, grads = 0.0, {}
        ############################################################################
        # TODO: Implement the backward pass for the fully connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch/layer normalization, you don't need to regularize the   #
        # scale and shift parameters.                                              #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        loss, dout = softmax_loss(scores, y)
        #print("loss before regularization",loss)
        '''
        # will automate the loss 
        loss +=0.5* self.reg*(np.sum(self.params['W1'] ** 2)+ \
                np.sum(self.params['W2'] ** 2)+np.sum(self.params['W3'] ** 2))
        
        '''
        # Automate the calculation of loss 

        l2_reg = 0.0
        for i in range(1,self.num_layers+1):
          
          l2_reg += np.sum(self.params['W{0}'.format(i)]**2)

        loss += 0.5 * self.reg * l2_reg

        #Backward pass

        grads = {}
        
        #1. Backward pass corresponding to no normalization and no dropouts
        if self.normalization == None:
          n = self.num_layers
          dout,grads['W{0}'.format(n)],grads['b{0}'.format(n)] = \
                            affine_backward(dout,cache_hist['cache{0}'.format(n)])
          
          for i in reversed(range(self.num_layers)):
           if i == 0:
             break
           else:
             dout, grads['W{0}'.format(i)],grads['b{0}'.format(i)]  \
                            = affine_relu_backward \
                            (dout,cache_hist['cache{0}'.format(i)])

        #2. Backward pass corresponding to no normalization and yes dropouts
        elif self.normalization == None and self.use_dropout:

          n = self.num_layers
          dout,grads['W{0}'.format(n)],grads['b{0}'.format(n)] = \
                            affine_backward(dout,cache_hist['cache{0}'.format(n)])
          
          for i in reversed(range(self.num_layers)):
           if i == 0:
             break
           else:
             dout, grads['W{0}'.format(i)],grads['b{0}'.format(i)]  \
                            = affine_relu_dropout_backward \
                            (dout,cache_hist['cache{0}'.format(i)])

        #3. Backward pass corresponding to batch normalization and no dropouts
        elif self.normalization == 'batchnorm':

          n = self.num_layers
          dout,grads['W{0}'.format(n)],grads['b{0}'.format(n)] = \
                            affine_backward(dout,cache_hist['cache{0}'.format(n)])
          
          for i in reversed(range(self.num_layers)):
           if i == 0:
             break
           else:
             dout, grads['W{0}'.format(i)],grads['b{0}'.format(i)], \
            grads['gamma{0}'.format(i)],grads['beta{0}'.format(i)]  \
                            = affine_batchnorm_relu_backward \
                            (dout,cache_hist['cache{0}'.format(i)])

        #4. Backward pass corresponding to layer normalization and no dropouts
        elif self.normalization == 'layernorm':

          n = self.num_layers
          dout,grads['W{0}'.format(n)],grads['b{0}'.format(n)] = \
                            affine_backward(dout,cache_hist['cache{0}'.format(n)])
          
          for i in reversed(range(self.num_layers)):
           if i == 0:
             break
           else:
             dout, grads['W{0}'.format(i)],grads['b{0}'.format(i)], \
            grads['gamma{0}'.format(i)],grads['beta{0}'.format(i)]  \
                            = affine_layernorm_relu_backward \
                            (dout,cache_hist['cache{0}'.format(i)])
    
        '''
        
        if self.normalization == None:
        
          dout,grads['W3'],grads['b3'] = affine_backward(dout, cache3) 
          dout,grads['W2'],grads['b2'] = affine_relu_backward(dout,cache2) 
          dout,grads['W1'],grads['b1'] = affine_relu_backward(dout,cache1)

        
        elif self.normalization == 'batchnorm':

          dout,grads['W8'],grads['b8'] = affine_backward(dout,cache8)
          dout,grads['W7'],grads['b7'],grads['gamma7'],grads['beta7'] = affine_batchnorm_relu_backward(dout, cache7) 
          dout,grads['W6'],grads['b6'],grads['gamma6'],grads['beta6'] = affine_batchnorm_relu_backward(dout, cache6)
          dout,grads['W5'],grads['b5'],grads['gamma5'],grads['beta5'] = affine_batchnorm_relu_backward(dout, cache5)
        
        '''
        # Adding the regulargization 

        # automate 
        

        for i in range(1,self.num_layers+1):
          
          grads['W{0}'.format(i)] += 0.5 * 2 *self.reg * self.params['W{0}'.format(i)]

    

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
