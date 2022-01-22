import numpy as np

def gradient_descent(D, x0, loss_f, grad_f, lr, tol, max_iter):
    losses = np.zeros(max_iter)
    y_old = x0
    y = x0
    for i in range(max_iter):
       
        g = grad_f(D, y)
        
        y = y_old - lr * g
        stress = loss_f(D, y)

        losses[i] = stress
        if stress < tol:
            msg = "\riter: {0}, stress: {1:}".format(i, stress)
            print(msg,flush=True,end="\t")
            losses = losses[:i]
            break
            
        if i % 50 == 0:
            msg = "\riter: {0}, stress: {1:}".format(i, stress)
            print(msg,flush=True,end="\t")
            
        y_old = y
        
    if i == max_iter-1:
        msg = "\riter: {0}, stress: {1:}".format(i, stress)
        print(msg,flush=True,end="\t")
        
    print('\n')

    return y, losses
