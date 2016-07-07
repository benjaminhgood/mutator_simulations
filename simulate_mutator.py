import numpy
from numpy.random import poisson
from math import log,exp
import sys

def evolve_mutator(N, s1, U1, s2, U2, r, mu, tmax, teq=None):
    
    if teq==None:
        teq = 20000  
    
    W1 = exp(s1)
    W2 = exp(s2)
    Um = 0
    sizes = numpy.array([[N,0.0],[0.0,0.0]])
    mutator_sizes = numpy.array([[0.0,0.0],[0.0,0.0]])
    fitnesses = numpy.array([[1.0,W1],[W2,W1*W2]])
    min_X1 = 0.0
    min_X2 = 0.0
    eq_X1 = None
    eq_X2 = None
    eq_X = None
        
    tfix = -1    
    for t in xrange(0,tmax+teq):
        
        if t==teq:
            eq_X1 = min_X1 + s1*(sizes*numpy.tile(numpy.arange(0,fitnesses.shape[1]),(fitnesses.shape[0],1))).sum()/sizes.sum()
            eq_X2 = min_X2 + s2*(sizes*numpy.tile(numpy.reshape(numpy.arange(0,fitnesses.shape[0]),(fitnesses.shape[0],1)),(1,fitnesses.shape[1]))).sum()/sizes.sum()
            eq_X = min_X1+min_X2+log((sizes*fitnesses).sum()/sizes.sum())
            Um = mu
            
        expected_sizes = sizes*fitnesses*(N/((sizes+mutator_sizes)*fitnesses).sum()) 
        expected_mutator_sizes = mutator_sizes*fitnesses*(N/((sizes+mutator_sizes)*fitnesses).sum()) 
        sizes = poisson((1-U1-U2-Um)*expected_sizes);
        sizes += numpy.roll(poisson(U1*expected_sizes),1,axis=1)
        sizes += numpy.roll(poisson(U2*expected_sizes),1,axis=0)
        
        mutator_sizes = poisson((1-r*(U1+U2))*expected_mutator_sizes)
        mutator_sizes += poisson(Um*expected_sizes);
        mutator_sizes += numpy.roll(poisson(r*U1*expected_mutator_sizes),1,axis=1)
        mutator_sizes += numpy.roll(poisson(r*U2*expected_mutator_sizes),1,axis=0)
        
        shrink_rows = ((sizes[0,:]+mutator_sizes[0,:]).sum() < 1)*1.0
        shrink_columns = ((sizes[:,0]+mutator_sizes[:,0]).sum() < 1)*1.0   
        expand_rows = ((sizes[-1,:]+mutator_sizes[-1,:]).sum() > 0)*1.0    
        expand_columns = ((sizes[:,-1]+mutator_sizes[:,-1]).sum() > 0)*1.0
        
        min_X1 += s1*shrink_columns
        min_X2 += s2*shrink_rows
        
        new_sizes = numpy.zeros((sizes.shape[0]+expand_rows-shrink_rows,sizes.shape[1]+expand_columns-shrink_columns))
        new_sizes[0:sizes.shape[0]-shrink_rows, 0: sizes.shape[1]-shrink_columns] = sizes[shrink_rows:,shrink_columns:]
        
        new_mutator_sizes = numpy.zeros((mutator_sizes.shape[0]+expand_rows-shrink_rows,mutator_sizes.shape[1]+expand_columns-shrink_columns))
        new_mutator_sizes[0:mutator_sizes.shape[0]-shrink_rows, 0: mutator_sizes.shape[1]-shrink_columns] = mutator_sizes[shrink_rows:,shrink_columns:]
        
        new_fitnesses = numpy.zeros_like(new_sizes)
        new_fitnesses[0:fitnesses.shape[0]-shrink_rows, 0: fitnesses.shape[1]-shrink_columns] = fitnesses[:fitnesses.shape[0]-shrink_rows,0:fitnesses.shape[1]-shrink_columns]
        new_fitnesses[-1,:] = new_fitnesses[-2,:]*W2
        new_fitnesses[:,-1] = new_fitnesses[:,-2]*W1
        sizes = new_sizes
        mutator_sizes = new_mutator_sizes
        fitnesses = new_fitnesses
        
        if sizes.sum() < 1:
            tfix = t-teq
            break
       
        
    v1 =  (min_X1 + s1*((sizes+mutator_sizes)*numpy.tile(numpy.arange(0,fitnesses.shape[1]),(fitnesses.shape[0],1))).sum()/(sizes+mutator_sizes).sum()-eq_X1)/tmax           
    v2 = (min_X2 + s2*((sizes+mutator_sizes)*numpy.tile(numpy.reshape(numpy.arange(0,fitnesses.shape[0]),(fitnesses.shape[0],1)),(1,fitnesses.shape[1]))).sum()/(sizes+mutator_sizes).sum() -eq_X2)/tmax
    
    v =  (min_X1+min_X2+log(((sizes+mutator_sizes)*fitnesses).sum())/((sizes+mutator_sizes).sum())-eq_X)/tmax

    return tfix, v1, v2, v
 
if __name__=='__main__':
    if len(sys.argv) < 8:
        print "Usage: python %s N s1 U1 s2 U2 r mu" % sys.argv[0]
    else:    
        N = float(sys.argv[1])
        s1 = float(sys.argv[2])
        U1 = float(sys.argv[3])
        s2 = float(sys.argv[4])
        U2 = float(sys.argv[5])
        r = float(sys.argv[6])
        R_target = float(sys.argv[7])
        mu = R_target

        
        print "# Modifier velocities:"
        for i in xrange(0,1):
            x, y, z,vm = evolve_mutator(N,s1,r*U1,s2,r*U2,r,0,50000)
            print vm


        print "# Wildtype velocities:"
        for i in xrange(0,10):
            x, y, z,v = evolve_mutator(N,s1,U1,s2,U2,r,0,50000)       
            print v


        print "# Mutator competition:"
        for i in xrange(0,60):
            tfix, x, y, z = evolve_mutator(N,s1,U1,s2,U2,r,mu,500000)
            print mu, tfix
            if tfix<1:
                tfix = 1000000 
            mu = min([1e-02, mu*(R_target*tfix)])
                        
