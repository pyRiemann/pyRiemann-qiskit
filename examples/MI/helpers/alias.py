from pyriemann.classification import MDM
from pyriemann.estimation import ERPCovariances
from sklearn.pipeline import make_pipeline

ERPCov_MDM = make_pipeline(ERPCovariances(estimator="lwf"), MDM())
