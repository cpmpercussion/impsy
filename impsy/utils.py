SIZE_TO_PARAMETERS = {
  'xxs': {
      "units": 16,
      "mixes": 5,
      "layers": 2,
  },
  'xs': {
      "units": 32,
      "mixes": 5,
      "layers": 2,
  },
  's': {
      "units": 64,
      "mixes": 5,
      "layers": 2
    },
  'm': {
      "units": 128,
      "mixes": 5,
      "layers": 2
    },
  'l': {
      "units": 256,
      "mixes": 5,
      "layers": 2
    },
  'xl': {
      "units": 512,
      "mixes": 5,
      "layers": 3
  },
  'default': {
    "units": 128,
    "mixes": 5,
    "layers": 2
  }
}

def mdrnn_config(size: str):
  """Get a config dictionary from a size string as used in the IMPS command line interface."""
  return SIZE_TO_PARAMETERS[size]
