import datetime
import numpy

from tensorflow.python.platform import gfile

def flatten(l):
    """ Merges a list of lists into a single list. """
    return [item for sublist in l for item in sublist]

def calculate_hit_at_one(predictions, actuals):
    """Performs a local (numpy) calculation of the hit at one.

    Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch'.

    Returns:
    float: The average hit at one across the entire batch.
    """
    top_prediction = numpy.argmax(predictions, 1)
    hits = [1 if (x == y) else 0 for x,y in zip(actuals, top_prediction)]
    return numpy.average(hits)

def calculate_hit_at_five(predictions, actuals):
    """Performs a local (numpy) calculation of the hit at one.

    Args:
    predictions: Matrix containing the outputs of the model.
      Dimensions are 'batch' x 'num_classes'.
    actuals: Matrix containing the ground truth labels.
      Dimensions are 'batch'.

    Returns:
    float: The average hit at five across the entire batch.
    """
    top_five_prediction = numpy.stack([numpy.argsort(predictions)[x][-5:] for x in range(predictions.shape[0])], 0)
    hits = [1 if (x in y) else 0 for x,y in zip(actuals, top_five_prediction)]
    return numpy.average(hits)

class EvaluationMetrics(object):
  """A class to store the evaluation metrics."""

  def __init__(self, num_class, top_k):
    """Construct an EvaluationMetrics object to store the evaluation metrics.

    Args:
      num_class: A positive integer specifying the number of classes.
      top_k: A positive integer specifying how many predictions are considered per video.

    Raises:
      ValueError: An error occurred when MeanAveragePrecisionCalculator cannot
        not be constructed.
    """
    self.sum_hit_at_one = 0.0
    self.sum_loss = 0.0
    self.top_k = top_k
    self.num_examples = 0

  def accumulate(self, predictions, labels, loss):
    """Accumulate the metrics calculated locally for this mini-batch.

    Args:
      predictions: A numpy matrix containing the outputs of the model.
        Dimensions are 'batch' x 'num_classes'.
      labels: A numpy matrix containing the ground truth labels.
        Dimensions are 'batch' x 'num_classes'.
      loss: A numpy array containing the loss for each sample.

    Returns:
      dictionary: A dictionary storing the metrics for the mini-batch.

    Raises:
      ValueError: An error occurred when the shape of predictions and actuals
        does not match.
    """
    batch_size = labels.shape[0]
    mean_hit_at_one = calculate_hit_at_one(predictions, labels)
    mean_loss = numpy.mean(loss)

    self.num_examples += batch_size
    self.sum_hit_at_one += mean_hit_at_one * batch_size
    self.sum_loss += mean_loss * batch_size

    return {"hit_at_one": mean_hit_at_one, "loss": mean_loss}

  def get(self):
    """Calculate the evaluation metrics for the whole epoch.

    Raises:
      ValueError: If no examples were accumulated.

    Returns:
      dictionary: a dictionary storing the evaluation metrics for the epoch. The
        dictionary has the fields: avg_hit_at_one, avg_perr, avg_loss, and
        aps (default nan).
    """
    if self.num_examples <= 0:
      raise ValueError("total_sample must be positive.")
    avg_hit_at_one = self.sum_hit_at_one / self.num_examples
    avg_loss = self.sum_loss / self.num_examples

    epoch_info_dict = {}
    return {"avg_hit_at_one": avg_hit_at_one, "avg_loss": avg_loss}

  def clear(self):
    """Clear the evaluation metrics and reset the EvaluationMetrics object."""
    self.sum_hit_at_one = 0.0
    self.sum_loss = 0.0
    self.num_examples = 0
