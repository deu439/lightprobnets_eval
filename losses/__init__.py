import losses.classification_losses
import losses.endpoint_error
import losses.probabilistic_classification_losses
import losses.probabilistic_endpoint_error


ClassificationLoss = classification_losses.ClassificationLoss
DirichletProbOutLoss = probabilistic_classification_losses.DirichletProbOutLoss
MultiScaleEPE = endpoint_error.MultiScaleEPE
MultiScaleLaplacian = probabilistic_endpoint_error.MultiScaleLaplacian
