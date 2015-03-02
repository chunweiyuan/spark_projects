import random
import numpy

class Kalman1D(object):
      # 1D implementation of the linear Kalman filter
      # A, B, H, Q, R are assumed 
      # Largely inspired by Welch & Bishop (2006), as well as greg.czerniak.info/guides/kalman1/
   
      def __init__(self, A, B, H, x0, P0, Q, R):
          self.A = A        # State transition matrix.
          self.B = B        # Control matrix.
          self.H = H        # Observation matrix.
          self.x = x0       # Initialize state estimate.
          self.P = P0       # Initialize covariance estimate.
          self.Q = Q        # Estimated error in process.
          self.R = R        # Estimated error in measurements.

      def current_state(self):
          return self.x[0,0], self.P[0,0]

      def time_update(self, u):
          # xnew = A*x + B*u, where u is the control vector
          # Pnew = A*P*A^T + Q
          self.x = self.A * self.x + self.B * u
          self.P = (self.A * self.P) * numpy.transpose(self.A) + self.Q

      def measurement_update(self, z):
          # z is the measurement vector (z = H*x + v, where v is measurement noise)
          # make sure self.x and self.p have already been updated via self.time_update
          # K = P * H^T * (H*P*H^T + R)^-1
          # x = x + K * (z - H*x)
          # P = (1 - K*H) * P
          innovation = z - (self.H * self.x)
          innovation_covariance = (self.H * self.P) * numpy.transpose(self.H) + self.R 
          K = (self.P * numpy.transpose(self.H)) * numpy.linalg.inv(innovation_covariance) # Kalman gain
          self.x += K * innovation
          self.P = (numpy.eye(self.P.shape[0]) - K * self.H) * self.P

      def update(self, u, z):
          self.time_update(u)
          self.measurement_update(z)


