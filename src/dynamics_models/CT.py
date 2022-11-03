import numpy as np

class CT():
    def __init__(self):
        self.omega = 2*np.pi/7
    
    
    
    def f(self, x, u, T):

        F = self.F(x, u, T)

        return F@x

    def F(self, x, u, T):

        F = np.eye(self.n*2)
        F[:self.n, self.n:] = np.array([[0, -self.omega], [self.omega, 0]])*T


        return F