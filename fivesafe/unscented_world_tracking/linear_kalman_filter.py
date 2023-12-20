import numpy as np

class LinearKalmanFilter(object):
    """Kalman Filer object which represents a linear kalman filter.


        German descriptions since the englisch word are not known to me - sorry - 

        :ivar A: Zustandsübergangsmatrix, beschreibt die Bewegung eines Zustandsvektors innerhalb eines Zeitschrittes
        
        :ivar B: Eingangsmatrix, beschreibt die Abhängkeit der Eingangsgrößen u auf den Zustandsvektor
        
        :ivar Q: Zustandsunsicherheit, beschreibt die Unsicherheiten eines Zustandsübergnags
        
        :ivar R: Messunsicherheit, beschreibt die Unsicherheit der Messung
        
        :ivar H: Messübergangsmatrix, beschreibt den Übergang einer Messung auf den Zustandsvektor 
        
        :ivar x_0: Initaler Zustand 
    """
    def __init__(self, A, B, Q, R, H, x_0, uncertainty_factor=1):
        """


        German descriptions since the englisch word are not known to me - sorry - 

        :param A: Zustandsübergangsmatrix, beschreibt die Bewegung eines Zustandsvektors innerhalb eines Zeitschrittes
        
        :param B: Eingangsmatrix, beschreibt die Abhängkeit der Eingangsgrößen u auf den Zustandsvektor
        
        :param Q: Zustandsunsicherheit, beschreibt die Unsicherheiten eines Zustandsübergnags
        
        :param R: Messunsicherheit, beschreibt die Unsicherheit der Messung
        
        :param H: Messübergangsmatrix, beschreibt den Übergang einer Messung auf den Zustandsvektor 
        
        :param x_0: Initaler Zustand 
        
        """



        self.A = A
        self.B = B

        self.H = H

        self.Q = Q


        self.R = R

        self.P = R * uncertainty_factor
        
        self.x = x_0




    def predict(self, u):
        """Calls the prediciton step of the kalman filter

        :param u: Input vector (Steuervektor)
        :return: State vector after prediction step.
        """


        # Update time state
        self.x = np.matmul(self.A, self.x) + np.matmul(self.B, u)

        # Calculate error covariance
        self.P = np.matmul(np.matmul(self.A, self.P), self.A.T) + self.Q
        return self.x

    def update(self, z):
        """Calls the update step by a given measurement

        :param z: Measurement vector (Messvektor)

        :return: State vector after update step.

        """

        # S = H*P*H'+R
        S = np.matmul(self.H, np.matmul(self.P, self.H.T)) + self.R

        # Calculate the Kalman Gain
        # K = P * H'* inv(H*P*H'+R)
        K = np.matmul(np.matmul(self.P, self.H.T), np.linalg.inv(S))  # Eq.(11)

        self.x = self.x + np.matmul(K, (z - np.matmul(self.H, self.x)))  # Eq.(12)

        I = np.eye(self.H.shape[1])
        self.P = (I - (K * self.H)) * self.P  # Eq.(13)

        return self.x 

    def get_state(self):
        """Getter for current state vector

        :return: State Vector
        
        """
        return self.x 


    def get_covariance_matrix_of_state(self):
        """Getter for current covariance matrice of state vector

        :return: covariance metrice
        
        """
        return self.P

def linear_paramter_interpolation(x0, y0, x1, y1, x):
    """
    Perform linear interpolation between two points (x0, y0) and (x1, y1) to find the value of y at a given x.

    Parameters:
    :param x0: The x-coordinate of the first point.
    :param y0: The y-coordinate of the first point.
    :param x1: The x-coordinate of the second point.
    :param y1: The y-coordinate of the second point.
    :param  x: The x-coordinate at which to interpolate.

    :return: The interpolated value of y. If x is outside the range [x0, x1], the function returns the corresponding y value of the closest boundary point.

    Note:
        This function does not extrapolate beyond the given boundaries of x.

    """
    if abs(x1 - x0) < 0.00001:
        raise ZeroDivisionError()    
    elif x <= x0:
        return y0
    elif x >= x1:
        return y1

    else:
        # Calculate the slope
        slope = (y1 - y0) / (x1 - x0)

        # Calculate the y-intercept
        y_intercept = y0 - slope * x0

        # Calculate the interpolated value of y
        y = slope * x + y_intercept

        return y
        