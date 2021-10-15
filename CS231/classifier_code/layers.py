from builtins import range
import numpy as np
import pandas as pd
import math

def affine_forward(x, w, b):
    """Computes the forward pass for an affine (fully connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # reshaping dimension of x from (N,d_1,....,d_k) to (N,M)
    arr = np.array(x.shape)
    c = np.prod(arr[1:])
    x_forward = x.reshape(arr[0],c)

    dot = np.dot(x_forward,w)

    out = np.array(dot[:,np.newaxis] + b).reshape(x_forward.shape[0],w.shape[1])


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """Computes the backward pass for an affine (fully connected) layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    # reshape x 
    arr = np.array(x.shape)
    c = np.prod(arr[1:])
    x_back = x.reshape(arr[0],c)
    
    dx = np.dot(dout,w.T).reshape(x.shape)
     
    x_back = x_back.T
    w_arr = []
    for i in range(0,w.shape[0]):
      val = np.dot(dout.T,x_back[i])
      w_arr.append(val)

    dw = np.array(w_arr).reshape(w.shape)

    db = np.sum(dout.T,axis=1)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def relu_forward(x):
    """Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    out = None
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    out = np.maximum(0,x)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = x
    return out, cache


def relu_backward(dout, cache):
    """Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache
    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    relu_out = np.maximum(0,x)
    relu_out[relu_out>0]=1
   
    dx = np.multiply(dout,relu_out)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    
    return dx


def softmax_loss(x, y):
    """Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    loss, dx = None, None

    ###########################################################################
    # TODO: Copy over your solution from Assignment 1.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    #print(type(x))
    N, D = x.shape
    loss = 0.0
    dx = np.zeros(x.shape)
    softmax_out = np.zeros(x.shape[0])
    norm_exp = np.array(np.max(x,axis=1)).reshape(x.shape[0],1)

    x = x - norm_exp
   
    sum_exp = np.sum(np.exp(x),axis = 1)

    # Softmax function loop 
    for i in range(0,x.shape[0]):
      
      # handling the math error exception
      if np.divide(np.exp(x[i,y[i]]),sum_exp[i]) == 0.0:
        softmax_out[i]= 0.0
      else:
        softmax_out[i] = - math.log(np.divide(np.exp(x[i,y[i]]),sum_exp[i]))
    
    # loss 
    loss = np.sum(softmax_out)
  
  
    #gradient
    for i in range(0,x.shape[0]):
      for j in range(0,x.shape[1]):
        if y[i] == j:

          y_hat = np.divide(np.exp(x[i,j]),sum_exp[i])
          dx[i,j] = y_hat*(1-y_hat)
   
        else:
          y_hat_1 = np.divide(np.exp(x[i,y[i]]),sum_exp[i])
          y_hat_2 = np.divide(np.exp(x[i,j]),sum_exp[i])
          dx[i,j] = - y_hat_1*y_hat_2 
      
        dx[i,j] *=  - np.divide(sum_exp[i],np.exp(x[i,y[i]]))
 
    loss /= N
 
    dx /= N

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return loss, dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    #1. calcuate the mean of the data
    mu = np.mean(x, axis=0)
    #2. calculate the variance of the data
    var = 1/float(N)* np.sum((x-mu) ** 2, axis=0)
    if mode == "train":
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #                                                                     #
        # Note that though you should be keeping track of the running         #
        # variance, you should normalize the data based on the standard       #
        # deviation (square root of variance) instead!                        #
        # Referencing the original paper (https://arxiv.org/abs/1502.03167)   #
        # might prove to be helpful.                                          #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        #3. normalize the data
        x_hat =  (x - mu)/ np.sqrt((var)+eps)
        #4. Use gamma and beta to scale and shift
        out = gamma*x_hat + beta
        
        #5. Update the running mean and variance as per the 
            #direction specified in the comments section.
        running_mean = momentum * running_mean + (1 - momentum) * mu
        running_var = momentum * running_var + (1 - momentum) * var
       
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        x_hat =  (x - running_mean)/ np.sqrt((running_var)+eps)
        out = gamma*x_hat + beta 

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var
    cache = (x,x_hat,mu,var,gamma,beta,eps)
    return out, cache


def batchnorm_backward(dout, cache):
    """Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    N, D = dout.shape
    x,x_hat,mu,var,gamma,beta,eps = cache
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    # Referencing the original paper (https://arxiv.org/abs/1502.03167)       #
    # might prove to be helpful.                                              #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    #beta
    dbeta = np.sum(dout,axis=0)
    #gamma
    dgamma = np.sum((dout*x_hat),axis=0)

    #dx
    """dl/dx = dl\dx` * dx`/dx + dl/dvar * dvar/dx + dl/du * du/dx """

    #1.dl\dx` * dx`/dx

    dx_hat = dout * gamma

    dbarx_x = 1/np.sqrt(var+eps)

    dx1 = dx_hat * dbarx_x

    #2. dl/dvar * dvar/dx

    """dl/dvar = dl/dy * dy/dx` * dx`/dvar"""

    dvar = (-1/2) * np.sum(dx_hat * (x - mu),axis=0)*((var + eps)**(-3/2))

    dvar_x = 2 * (1/N) * (x-mu) * np.ones(x.shape)

    dx2 = dvar * dvar_x

    #3.dl/du * du/dx

    """dl/du = dl/dx` * dx`/ du + dl/dvar * dvar/du"""
    

    dmu1 = np.sum(dx_hat,0) * (-1) * (1/np.sqrt(var+eps))

    dmu2= (-2)  * dvar * np.sum((x - mu),axis=0)

    dmu_x = (1/N) * np.ones(x.shape)

    dx3 = (dmu1 + dmu2) * dmu_x

    dx = dx1 + dx2 + dx3


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.
    See the jupyter notebook for more hints.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def layernorm_forward(x, gamma, beta, ln_param):
    """Forward pass for layer normalization.

    During both training and test-time, the incoming data is normalized per data-point,
    before being scaled by gamma and beta parameters identical to that of batch normalization.

    Note that in contrast to batch normalization, the behavior during train and test-time for
    layer normalization are identical, and we do not need to keep track of running averages
    of any sort.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - ln_param: Dictionary with the following keys:
        - eps: Constant for numeric stability

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    out, cache = None, None
    eps = ln_param.get("eps", 1e-5)

    ###########################################################################
    # TODO: Implement the training-time forward pass for layer norm.          #
    # Normalize the incoming data, and scale and  shift the normalized data   #
    #  using gamma and beta.                                                  #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of  batch normalization, and inserting a line or two of  #
    # well-placed code. In particular, can you think of any matrix            #
    # transformations you could perform, that would enable you to copy over   #
    # the batch norm code and leave it almost unchanged?                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    x = x.T
    N,D = x.shape

    mu = np.mean(x, axis=0)
    var = 1/float(N)* np.sum((x-mu) ** 2, axis=0)

    x_hat =  (x - mu)/ np.sqrt((var)+eps)

    x_hat = np.array(x_hat.T)
    
    #4. Use gamma and beta to scale and shift
    out = gamma*x_hat + beta

    x = np.array(x.T)


    cache = (x,x_hat,mu,var,gamma,beta,eps)

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def layernorm_backward(dout, cache):
    """Backward pass for layer normalization.

    For this implementation, you can heavily rely on the work you've done already
    for batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from layernorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    dx, dgamma, dbeta = None, None, None
    
    
    x,x_hat,mu,var,gamma,beta,eps = cache
    
    ###########################################################################
    # TODO: Implement the backward pass for layer norm.                       #
    #                                                                         #
    # HINT: this can be done by slightly modifying your training-time         #
    # implementation of batch normalization. The hints to the forward pass    #
    # still apply!                                                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    #beta
    dbeta = np.sum(dout,axis=0)
    
    #gamma
    dgamma = np.sum((dout*x_hat),axis=0)
    

    x_hat = x_hat.T
    x = x.T
    N, D = x_hat.shape
  
    #dx
    """dl/dx = dl\dx` * dx`/dx + dl/dvar * dvar/dx + dl/du * du/dx """

    #1.dl\dx` * dx`/dx

    dx_hat = dout * gamma

    dx_hat = dx_hat.T

    dbarx_x = 1/np.sqrt(var+eps)

    dx1 = dx_hat * dbarx_x

    #2. dl/dvar * dvar/dx

    """dl/dvar = dl/dy * dy/dx` * dx`/dvar"""

    dvar = (-1/2) * np.sum(dx_hat * (x - mu),axis=0)*((var + eps)**(-3/2))

    dvar_x = 2 * (1/N) * (x-mu) * np.ones(x.shape)

    dx2 = dvar * dvar_x

    #3.dl/du * du/dx

    """dl/du = dl/dx` * dx`/ du + dl/dvar * dvar/du"""
    

    dmu1 = np.sum(dx_hat,0) * (-1) * (1/np.sqrt(var+eps))

    dmu2= (-2)  * dvar * np.sum((x - mu),axis=0)

    dmu_x = (1/N) * np.ones(x.shape)

    dx3 = (dmu1 + dmu2) * dmu_x

    dx = dx1 + dx2 + dx3

    dx = dx.T


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """Forward pass for inverted dropout.

    Note that this is different from the vanilla version of dropout.
    Here, p is the probability of keeping a neuron output, as opposed to
    the probability of dropping a neuron output.
    See http://cs231n.github.io/neural-networks-2/#reg for more details.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We keep each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param["p"], dropout_param["mode"]
    if "seed" in dropout_param:
        np.random.seed(dropout_param["seed"])

    mask = None
    out = None

    if mode == "train":
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        mask = (np.random.rand(*x.shape) < p) / p # first dropout mask. Notice /p!
        out = x*mask # drop!
        

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == "test":
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        out = x


        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """Backward pass for inverted dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param["mode"]

    dx = None
    if mode == "train":
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        pass
        dx = dout*mask

        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == "test":
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width WW.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    During padding, 'pad' zeros should be placed symmetrically (i.e equally on both sides)
    along the height and width axes of the input. Be careful not to modfiy the original
    input x directly.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    HH = np.int(1 + (x.shape[2] + 2 * conv_param['pad'] - w.shape[2]) / conv_param['stride'])
    WW = np.int(1 + (x.shape[3] + 2 * conv_param['pad'] - w.shape[3]) / conv_param['stride'])
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
   
    lst = []
    x_params =  {}

    # The number of images passed through the network. Basically N
    for i in range(0,x.shape[0]):
     
     # Creating another loop for the depth which is F

      for j in range(0,x.shape[1]):

      # Now we add 0-pad to each 2D phase of the image, and store the
      # extracted portions with zero padding.
  
        x_params['x{0}'.format(j)]= np.pad(x[i,j,:,:],pad_width=1)
        
      height =np.int((x_params['x0'].shape[0] - w.shape[2])/conv_param['stride'] + 1)
      width = np.int((x_params['x0'].shape[1] - w.shape[3])/conv_param['stride'] + 1)
    
    # Now performing the operation of convolution on x_params
    # for loop to loop across all the filters, in this case, its F=3
      for e in range(0,w.shape[0]):

        # for loop to traverse the width of the image.
        for p in range(0,width):

          p = p*conv_param['stride']
            
          for q in range(0,height):

            q = q*conv_param['stride'] 

            prod_sum = 0
            for k in range(0,len(x_params.keys())):
            
              prod_sum += \
                np.sum(x_params['x{0}'.format(k)][p:p+w.shape[3],q:q+w.shape[2]]\
                * w[e,k,:,:])
           
            prod_sum += b[e,]
            lst.append(prod_sum)
  
    out = np.array(lst).reshape(x.shape[0],w.shape[0],HH,WW)


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """A naive implementation of the forward pass for a max-pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    No padding is necessary here, eg you can assume:
      - (H - pool_height) % stride == 0
      - (W - pool_width) % stride == 0

    Returns a tuple of:
    - out: Output data, of shape (N, C, H', W') where H' and W' are given by
      H' = 1 + (H - pool_height) / stride
      W' = 1 + (W - pool_width) / stride
    - cache: (x, pool_param)
    """
    out = None
    HH = np.int(1 + (x.shape[2] - pool_param['pool_height'])/pool_param['stride'])
    WW = np.int(1 + (x.shape[3] - pool_param['pool_width'])/pool_param['stride'])
    ###########################################################################
    # TODO: Implement the max-pooling forward pass                            #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    max_val_lst = []
    max_loc_lst = []
    x_params =  {}

    width_range = np.int(x.shape[3]/pool_param['pool_width'])
    height_range = np.int(x.shape[2]/pool_param['pool_height'])
    # The number of images passed through the network. Basically N
    for i in range(0,x.shape[0]):
     
     # Creating another loop for the depth which is F

      for j in range(0,x.shape[1]):

      # Now we add 0-pad to each 2D phase of the image, and store the
      # extracted portions with zero padding.
  
        x_params['x{0}'.format(j)]= x[i,j,:,:]
  
    # Now performing the operation of pooling on x_params

      for k in range(0,len(x_params.keys())):

        for p in range(0,width_range):

          p = p * pool_param['stride']
          
          for q in range(0,height_range):

            q = q * pool_param['stride'] 
          
            max_val = np.max(x_params['x{0}'.format(k)][p:p+pool_param['pool_width'],\
              q:q+pool_param['pool_height']])

            max_loc = np.argmax(x_params['x{0}'.format(k)][p:p+pool_param['pool_width'],\
              q:q+pool_param['pool_height']])
          
            max_val_lst.append(max_val)
            max_loc_lst.append(max_loc)
          
    
    out = np.array(max_val_lst).reshape(x.shape[0],x.shape[1],HH,WW)
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param,max_loc_lst)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """A naive implementation of the backward pass for a max-pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    #print("pool_params",cache)
    x,pool_param,max_loc_lst = cache
    pl_wd = pool_param['pool_width']
    pl_ht = pool_param['pool_height']
    width_range = np.int(x.shape[3]/pool_param['pool_width'])
    height_range = np.int(x.shape[2]/pool_param['pool_height'])
    ###########################################################################
    # TODO: Implement the max-pooling backward pass                           #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
  
    counter = 0
    dx_params = {}
    dout_params = {}
    c = 0 
    for i in range(0,x.shape[0]):
      
      for j in range(0,x.shape[1]):

        dx_params['dx_df{0}'.format(c)] = pd.DataFrame(np.zeros((x.shape[-1],x.shape[-2])))
        c +=1
      
    # so now we have all the 6 dataframes.

    # convert dout output also into 2D dataframes. 
    c = 0
    
    for i in range(0,x.shape[0]):
      
      for j in range(0,x.shape[1]):

        dout_params['dout_df{0}'.format(c)] = pd.DataFrame(dout[i,j,:,:])
        c +=1

    for k in range(0,c):

        for ww in range(0,width_range):

          w = ww * pool_param['stride']
          
          for hh in range(0,height_range):

            h = hh * pool_param['stride'] 

            pool_hw_array = np.zeros([np.prod(pl_wd*pl_ht),1])      

            pool_hw_array[max_loc_lst[counter]] = \
                    np.array(dout_params['dout_df{0}'.format(k)].iloc[ww:ww+1,hh:hh+1])
            
            pool_hw_array = pool_hw_array.reshape(pl_wd,pl_ht)

            dx_params['dx_df{0}'.format(k)].iloc[w:w+pl_wd,h:h+pl_ht] = pool_hw_array

            counter +=1
    
    # concatenating all the updated dataframes
    
    df = pd.concat(dx_params.values(), ignore_index=True)
    
    dx = np.array(df).reshape(x.shape)
    

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    N,C,H,W = x.shape
    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass

    """https://www.reddit.com/r/cs231n/comments/443y2g/hints_for_a2/"""

    x_hat = np.transpose(x,(0,2,3,1))

    N,H,W,C = x_hat.shape

    x_hat = x_hat.reshape(N*H*W,C)

    out,cache = batchnorm_forward(x_hat,gamma,beta, bn_param)

    out = np.reshape(out,(N,H,W,C))

    out = np.transpose(out,(0,3,1,2))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    #cache = (x,,mu,var,gamma,beta,eps)
    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None
    
    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization by calling the      #
    # vanilla version of batch normalization you implemented above.           #
    # Your implementation should be very short; ours is less than five lines. #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    
    dout_hat = np.transpose(dout,(0,2,3,1))
    N,H,W,C = dout_hat.shape
    dout_hat = np.reshape(dout_hat,(N*H*W,C))
    dx, dgamma, dbeta = batchnorm_backward(dout_hat,cache)

    dx = np.reshape(dx,(N,H,W,C))
    dx = np.transpose(dx,(0,3,1,2))
    
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def spatial_groupnorm_forward(x, gamma, beta, G, gn_param):
    """Computes the forward pass for spatial group normalization.
    
    In contrast to layer normalization, group normalization splits each entry in the data into G
    contiguous pieces, which it then normalizes independently. Per-feature shifting and scaling
    are then applied to the data, in a manner identical to that of batch normalization and layer
    normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (1, C, 1, 1)
    - beta: Shift parameter, of shape (1, C, 1, 1)
    - G: Integer mumber of groups to split into, should be a divisor of C
    - gn_param: Dictionary with the following keys:
      - eps: Constant for numeric stability

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None
    eps = gn_param.get("eps", 1e-5)
    N,C,H,W = x.shape
    ###########################################################################
    # TODO: Implement the forward pass for spatial group normalization.       #
    # This will be extremely similar to the layer norm implementation.        #
    # In particular, think about how you could transform the matrix so that   #
    # the bulk of the code is similar to both train-time batch normalization  #
    # and layer normalization!                                                #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    #"https://arxiv.org/pdf/1803.08494.pdf"
   
    batch = C//G
    
    gammat = np.reshape(gamma,(C,))
    betat  = np.reshape(beta,(C,))
    out = {}
    cache = {}
    for i in range(1,batch):

      r = x[:,(i-1)*batch:i*batch,:,:] 
      N,T,H,W = r.shape
      r = np.reshape(r,(N*H*W,T))
    
    
      out['out{0}'.format(i)],cache['cache{0}'.format(i)] = \
                    layernorm_forward(r,gammat[(i-1)*batch:i*batch,],\
                    betat[(i-1)*batch:i*batch],gn_param)
      
    cache['G'] = G
    cache['gamma'] = gamma
    cache['beta'] = beta
    out = np.hstack(list(out.values())).reshape(x.shape)
    cache = cache
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return out, cache


def spatial_groupnorm_backward(dout, cache):
    """Computes the backward pass for spatial group normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (1, C, 1, 1)
    - dbeta: Gradient with respect to shift parameter, of shape (1, C, 1, 1)
    """
    dx, dgamma, dbeta = None, None, None
    G = cache['G']
   
    N,C,H,W = dout.shape
    ###########################################################################
    # TODO: Implement the backward pass for spatial group normalization.      #
    # This will be extremely similar to the layer norm implementation.        #
    ###########################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    pass
    batch = C//G
  
    dout_hat = np.transpose(dout,(0,2,3,1))
    
    # We will do trial and error now 
    
    dx = {}
    dgamma = {}
    dbeta  = {}

    for i in range(1,batch):
      
      r = dout_hat[:,:,:,(i-1)*batch:i*batch] 
      N,H,W,T = r.shape
      r = np.reshape(r,(N*H*W,T))
      
    
      dx['dx{0}'.format(i)],dgamma['dgamma{0}'.format(i)],\
       dbeta['dbeta{0}'.format(i)]= \
                    layernorm_backward(r,cache['cache{0}'.format(i)])
      
    
    dx     = np.hstack(list(dx.values())).reshape(dout.shape)
    dgamma = np.hstack(list(dgamma.values())).reshape(cache['gamma'].shape)
    dbeta  = np.hstack(list(dbeta.values())).reshape(cache['beta'].shape)

   
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dgamma, dbeta
