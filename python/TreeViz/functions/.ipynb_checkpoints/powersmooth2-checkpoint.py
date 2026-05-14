import numpy as np
import numpy.matlib as mlib

def powersmooth2(vec,order,weight):
# BM Friedrich, 19.12.2014

# This is a modified version, which is easier to understand. For the original version please check:
#  https://www.mathworks.com/matlabcentral/fileexchange/48799-powersmooth?s_tid=srchtitle

# cost function to be minimized
# cost = @(vec,vecs) sum( (vec-vecs).^2 ) + ...
#                     weight*sum( diff(vecs,order).^2 );
# rewriting cost function as quadratic form: vecs.A.vecs+vecs.b+c->min

# An Gong, last modified 2021.04.21.

    N=len(vec);
    vec_smooth=np.full((N,1), np.nan)
    # isRow=isrow(vec);
    # if isRow
    #   vec = vec';
    # end
    if vec.ndim == 2 and vec.shape[1] == 1:
        vec =  vec
    else:
        vec.reshape(-1, 1)
    goodIndex = np.argwhere(vec==vec) #notice that np.NaN !=np.NaN, this is for removing NaN
    # goodIndex = goodIndex.reshape(len(goodIndex))
    goodIndex = goodIndex[:,0]
    vec = vec[goodIndex];

    n = len(vec);
    D = np.matlib.eye(n)-np.matlib.eye(n,n,-1); # D*vec is the first-order forward difference of vec plus an extral vec(1)
    #remove the first n points
    ek = np.zeros((n,n))
    for i in range(order, n):
        ek[i,i] = 1
    Ek = mlib.matrix(ek)
    Dk = Ek*(D**order);
    A = mlib.eye(n)+weight*(Dk.transpose())*Dk
    B = np.linalg.pinv(A)
    #this factor can be derived by taking the derivate of the cost function
    vec = vec.reshape((vec.size, 1))
    v = np.dot(B, vec)
    vec_smooth[goodIndex]=v;
    return vec_smooth
    # if isRow
    #     vecs=vecs';
    # end