import numpy as np
import init
from hdmm import matrix
from functools import reduce


class MemMatrix():
    '''
    support large matrices
    if multiple matrices X0,X1,...Xn are given,
    this matrix corresponds to the product of all the matrices: X0 * X1 * ... *Xn
    '''
    def __init__(self, X, is_memmap=False, dtype=np.float64, chunk=1000000, filename='temp/memmatrix.dat'):
        
        self.chunk=chunk
        self.filename=filename
        self.dtype = dtype

        if isinstance(X, list):
            self.matrices = X
            self.shape = (X[0].shape[0],X[-1].shape[1])
        else:
            
            self.matrices = [X]
            self.shape = X.shape
            
        self.memmap = is_memmap

        
    
    def explicit_matrix(self):
        X = [matrix for matrix in self.matrices]
        if len(X) == 1:
            return X[0]
        else:
            if self.memmap:
                # right associative: (X0*(X1...(Xn-1*Xn)))
                V = X[-1]
                Q = X[:-1]
                for Q_i in Q[::-1]:  # reverse the order
                    matmat(Q_i, V, chunk=self.chunk)
                    L = np.memmap('temp/matmat.dat', dtype=self.dtype, mode='r+',shape=(Q_i.shape[0],V.shape[1]))
                    V = np.memmap('temp/matmat_out.dat', dtype=self.dtype, mode='w+',shape=L.shape)
                    for i in range(0,L.shape[0],self.chunk):
                        j = min(i+self.chunk,L.shape[0])
                        V[i:j,::] = L[i:j,::]
                    del L
                return V
            else:
                return reduce(lambda A, B: B.dot(A), X[::-1]) # right asscociative
                
    
    def dot(self, Y):
        # implicitly represent matrix instead of computing the product
        assert isinstance(Y, MemMatrix)
        X = [matrix for matrix in self.matrices]
        for matrix in Y.matrices:
            X.append(matrix)
        return MemMatrix(X, is_memmap=Y.memmap,filename=Y.filename)

    @staticmethod    
    def laplace_noise(prng=np.random.mtrand._rand, scale=1.0, size=(1,1), filename='temp/memmatrix.dat',chunk=1000000):
        # generate a large size of laplacian noise with memmap
        #assert isinstance(size, int)
        assert len(size) == 2
        V = np.memmap(filename,mode='w+',dtype=np.float64,shape=size)
        for i in range(0,size[0],chunk):
            j = min(i+chunk,size[0])
            V[i:j,::] = prng.laplace(loc=0.0, scale=scale, size=(j-i, size[1]))
        V.flush()
        return MemMatrix(V, is_memmap=True, filename=filename)
        

def _matmat_memmap(Q, V, chunk=1000000, dtype=np.float64):
    '''
    Q: Matrix
    V: np.memmap
    '''
    
    assert isinstance(V, np.memmap)
    X = np.memmap("temp/matmat_memmap.dat",dtype=dtype,mode='w+',shape=(Q.shape[0],V.shape[1]))
    for i in range(0,V.shape[1],chunk):
        j = min(i+chunk,V.shape[1])
        X[::,i:j] = Q.dot(V[::,i:j])
    del X


def matmat(Q,V,chunk=1000000,dtype=np.float64):
    '''
    Q: VStack
    V: np.memmap
    '''
    assert isinstance(V, np.memmap)

    Y = np.memmap("temp/matmat.dat", dtype=dtype, mode='w+',shape=(Q.shape[0],V.shape[1]))
    if isinstance(Q, matrix.VStack):
        pos =0
        for Q_k in Q.matrices:
            _matmat_memmap(Q_k, V, chunk=chunk)
            X_k = np.memmap("temp/matmat_memmap.dat", dtype=dtype, mode='r+',shape=(Q_k.shape[0],V.shape[1]))
            for i in range(0, X_k.shape[1], chunk):
                j = min(i+chunk,X_k.shape[1])
                Y[pos:pos+X_k.shape[0],i:j] = X_k[::,i:j]
            pos+=X_k.shape[0]
            del X_k
        del Y

    else:
        _matmat_memmap(Q, V, chunk=chunk)
        X = np.memmap("temp/matmat_memmap.dat",dtype=dtype, mode='r+',shape=(Q.shape[0],V.shape[1]))
        for i in range(0, V.shape[1], chunk):
            j = min(i+chunk,X.shape[1])
            Y[::,i:j] = X[::,i:j]
        Y.flush()
        X.flush()
        del Y
        del X
