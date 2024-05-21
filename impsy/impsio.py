"""impsy.impsio: IO classes for interactions over OSC, Websockets, MIDI and Serial"""
import abc

class IOServer(abc.ABC):
  """Abstract class for music IO for IMPSY."""

  def __init__(self, config, callback) -> None:
    self.config = config # the IMPSY config
    self.callback = callback # a callback method to report incoming data.
  
  @abc.abstractmethod
  def send(self, value):
    pass




# class MIDIServer(IOServer):



# class OSCServer(IOServer):

