import numpy as np
from hmmlearn import hmm


startprob = np.array([0.0, 0.1, 0.8, 0.1, 0.0])
transmat = np.array([[0.9, 0.1, 0.0, 0.0, 0.0],
                     [0.4, 0.3, 0.3, 0.0, 0.0],
                     [0.1, 0.2, 0.4, 0.3, 0.0],
                     [0.0, 0.1, 0.3, 0.3, 0.3],
                     [0.0, 0.0, 0.3, 0.1, 0.6]])

means = np.array([[1.0, 1.0],
                  [2.0, 2.0],
                  [3.0, 3.0],
                  [4.0, 4.0],
                  [5.0, 5.0]])

covars = 0.1 * np.tile(np.identity(2), (5, 1, 1))

model = hmm.GaussianHMM(n_components=5, covariance_type="full")

model.startprob_ = startprob
model.transmat_ = transmat
model.means_ = means
model.covars_ = covars

X, Z = model.sample(10)
print(X)
print(Z)
