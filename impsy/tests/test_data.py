from impsy import dataset
import numpy as np
import os


def test_log_to_examples():
  """Tests transform_log_to_sequence_example with a single example"""
  test_log_file = "logs/2024-06-01T12-00-00-4d-mdrnn.log"
  test_line = "2024-06-01T12:00:00,interface,0.1,0.2,0.3,0.4"
  with open(test_log_file, "w") as file:
    file.write(test_line)
  log = dataset.transform_log_to_sequence_example(test_log_file, 4)
  assert(isinstance(log, np.ndarray))
  os.remove(test_log_file)


def test_dataset_command():
  """Test the dataset command runs"""
  print("Testing dataset function...")
  dataset.generate_dataset(dimension=2, source="logs")
