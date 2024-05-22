"""impsy.impsio: IO classes for interactions over OSC, Websockets, MIDI and Serial"""
import abc
from collections.abc import Callable
import click
from typing import List
import datetime
import numpy as np
import serial
import mido
from websockets.sync.server import serve
from threading import Thread


class IOServer(abc.ABC):
  """Abstract class for music IO for IMPSY."""

  config: dict
  callback: Callable[[int, float], None]

  def __init__(self, config, callback) -> None:
    self.config = config # the IMPSY config
    self.callback = callback # a callback method to report incoming data.
  
  @abc.abstractmethod
  def send(self, output_values) -> None:
    pass

  @abc.abstractmethod
  def connect(self) -> None:
    pass

  @abc.abstractmethod
  def disconnect(self) -> None:
      pass
  

class WebSocketServer(IOServer):
    """Handles Websocket Serving for IMPSY"""

    def __init__(self, config, callback) -> None:
        super().__init__(config, callback)
        self.ws_clients = set() # storage for potential ws clients.
        self.ws_thread = None

    def send(self, output_values) -> None:
        return super().send(output_values)
    
    def connect(self) -> None:
        click.secho("Preparing websocket thread.", fg='yellow')
        self.ws_thread = Thread(target=self.websocket_serve_loop, name="ws_receiver_thread", daemon=True)
        self.ws_thread.start() # send it!

    def disconnect(self) -> None:
        try:
          self.ws_thread.join(timeout=0.1)
        except:
          pass

    def websocket_send_midi(self, message):
        """Sends a mido MIDI message via websockets if available."""
        if message.type == "note_on":
            ws_msg = f"/channel/{message.channel}/noteon/{message.note}/{message.velocity}"
        elif message.type == "note_off":
            ws_msg = f"/channel/{message.channel}/noteoff/{message.note}/{message.velocity}"
        elif message.type == "control_change":
            ws_msg = f"/channel/{message.channel}/cc/{message.control}/{message.value}"
        else:
            return
        # click.secho(f"WS out: {ws_msg}")
        # Broadcast the ws_msg to all clients (sync version can't use websockets.broadcast function so doing this naively)
        for ws_client in self.ws_clients.copy():
            try:
                ws_client.send(ws_msg)
            except:
                self.ws_clients.remove(ws_client)

    def websocket_handler(self, websocket):
        """Handle websocket input messages that might arrive"""
        self.ws_clients.add(websocket) # add websocket to the client list.
        # do the actual handling
        for message in websocket:
            click.secho(f"WS: {message}", fg="red") # TODO: fine for debug, but should be removed really.
            m = message.split('/')[1:]
            msg_type = m[2]
            chan = int(m[1]) # TODO: should this be chan+1 or -1 or something.
            note = int(m[3])
            vel = int(m[4])
            if msg_type == "noteon":
                # note_on
                try:
                    index = self.config["midi"]["input"].index(["note_on", chan])
                    value = note / 127.0
                    self.callback(index, value)
                except ValueError:
                    click.secho(f"WS in: exception with message {message}", fg="red")
                    pass
            elif msg_type == "cc":
                # cc
                try:
                    index = self.config["midi"]["input"].index(["control_change", chan, note])
                    value = vel / 127.0
                    self.callback(index, value)
                except ValueError:
                    click.secho(f"WS in: exception with message {message}", fg="red")
                    pass
            # global websocket
            # ws_msg = f"/channel/{message.channel}/noteon/{message.note}/{message.velocity}"
            # ws_msg = f"/channel/{message.channel}/noteoff/{message.note}/{message.velocity}"
            # ws_msg = f"/channel/{message.channel}/cc/{message.control}/{message.value}"

    def websocket_serve_loop(self):
        """Threading websockets server following https://websockets.readthedocs.io/en/stable/reference/sync/server.html"""
        hostname = self.config['websocket']['server_ip']
        port = self.config['websocket']['server_port']
        with serve(self.websocket_handler, hostname, port) as server:
            server.serve_forever()

class OSCServer(IOServer):
    """Handles OSC IO for IMPSY."""

    def __init__(self, config, callback) -> None:
        super().__init__(config, callback)

    def connect(self) -> None:
        return super().connect()
    
    def disconnect(self) -> None:
        return super().disconnect()
    
    def send(self, output_values) -> None:
        return super().send(output_values)
    


class MIDIServer(IOServer):
    """Handles MIDI IO for IMPSY."""
    
    # midi_in_port: mido.ports.BaseInput
    # midi_in_port: mido.ports.BaseOutput

    def match_midi_port_to_list(port, port_list):
      """Return the closest actual MIDI port name given a partial match and a list."""
      if port in port_list:
          return port
      contains_list = [x for x in port_list if port in x]
      if not contains_list:
          return False
      else:
          return contains_list[0]

    def open_raspberry_serial():
        """Tries to open a serial port for MIDI IO on Raspberry Pi."""
        try:
            click.secho("Trying to open Raspberry Pi serial port for MIDI in/out.", fg='yellow')
            ser = serial.Serial('/dev/ttyAMA0', baudrate=31250)
        except:
            ser = None
            click.secho("Could not open serial port, might be in development mode.", fg='red')
        return ser
    
    def __init__(self, config, callback) -> None:
        super().__init__(config, callback)
        self.dimension = self.config["model"]["dimension"] # retrieve dimension from the config file.
        self.verbose = self.config["verbose"]
        self.last_midi_notes = {} # dict to store last played notes via midi
        self.websocket_send_midi = None # TODO implement some kind generic MIDI callback for other output channels.

    def connect(self) -> None:
        # Try Raspberry Pi serial opening
        self.serial = MIDIServer.open_raspberry_serial()
        # MIDI port opening
        click.secho("Opening MIDI port for input/output.", fg='yellow')
        try:
            desired_input_port = MIDIServer.match_midi_port_to_list(self.config["midi"]["in_device"], mido.get_input_names())
            self.midi_in_port = mido.open_input(desired_input_port)
            click.secho(f"MIDI: in port is: {self.midi_in_port.name}", fg='green')
        except: 
            self.midi_in_port = None
            click.secho("Could not open MIDI input.", fg='red')
            click.secho(f"MIDI Inputs: {mido.get_input_names()}", fg = 'blue')
        try:
            desired_output_port = MIDIServer.match_midi_port_to_list(self.config["midi"]["out_device"], mido.get_output_names())
            self.midi_out_port = mido.open_output(desired_output_port)
            click.secho(f"MIDI: out port is: {self.midi_out_port.name}", fg='green')
        except:
            self.midi_out_port = None
            click.secho("Could not open MIDI output.", fg='red')
            click.secho(f"MIDI Outputs: {mido.get_output_names()}", fg = 'blue')

    def disconnect(self) -> None:
      self.send_midi_note_offs()
      try:
        self.midi_in_port.close()
      except:
        pass
      try:
        self.midi_out_port.close()
      except:
        pass
      try:
        self.serial.close()
      except:
        pass

    def serial_send_midi(self, message):
        """Sends a mido MIDI message via the very basic serial output on Raspberry Pi GPIO."""
        try:
            self.serial.write(message.bin())
        except: 
            pass

    def send_midi_message(self, message):
        """Send a MIDI message across all required outputs"""
        # TODO: this is where we can have laggy performance, careful.
        if self.midi_out_port is not None:
            self.midi_out_port.send(message)
        self.serial_send_midi(message)
        if self.websocket_send_midi is not None:
          self.websocket_send_midi(message)

    def send_midi_note_on(self, channel, pitch, velocity):
        """Send a MIDI note on (and implicitly handle note_off)"""
        # stop the previous note
        try:
            midi_msg = mido.Message('note_off', channel=channel, note=self.last_midi_notes[channel], velocity=0)
            self.send_midi_message(midi_msg)
            # click.secho(f"MIDI: note_off: {self.last_midi_notes[channel]}: msg: {midi_msg.bin()}", fg="blue")
            # do this by whatever other channels necessary
        except KeyError:
            click.secho("Something wrong with turning MIDI notes off!!", fg="red")
            pass

        # play the present note
        midi_msg = mido.Message('note_on', channel=channel, note=pitch, velocity=velocity)
        self.send_midi_message(midi_msg)
        # click.secho(f"MIDI: note_on: {pitch}: msg: {midi_msg.bin()}", fg="blue")
        self.last_midi_notes[channel] = pitch

    def send_control_change(self, channel, control, value):
        """Send a MIDI control change message"""
        midi_msg = mido.Message('control_change', channel=channel, control=control, value=value)
        self.send_midi_message(midi_msg)

    def send_midi_note_offs(self):
        """Sends note offs on any MIDI channels that have been used for notes."""
        outconf = self.config["midi"]["output"]
        out_channels = [x[1] for x in outconf if x[0] == "note_on"]
        for i in out_channels:
            try:
                midi_msg = mido.Message('note_off', channel=i-1, note=self.last_midi_notes[i-1], velocity=0)
                self.send_midi_message(midi_msg)
                # click.secho(f"MIDI: note_off: {self.last_midi_notes[i-1]}: msg: {midi_msg.bin()}", fg="blue")
            except KeyError:
                click.secho("Something wrong with all MIDI Note off!", fg="red")
                pass

    def send(self, output_values):
        """Sends sound commands via MIDI"""
        assert len(output_values)+1 == self.dimension, "Dimension not same as prediction size." # Todo more useful error.
        start_time = datetime.datetime.now()
        outconf = self.config["midi"]["output"]
        values = list(map(int, (np.ceil(output_values * 127))))
        if self.verbose:
            click.secho(f'out: {values}', fg='green')
        for i in range(self.dimension-1):
            if outconf[i][0] == "note_on":
                self.send_midi_note_on(outconf[i][1]-1, values[i], 127) # note decremented channel (0-15)
            if outconf[i][0] == "control_change":
                self.send_control_change(outconf[i][1]-1, outconf[i][2], values[i]) # note decrement channel (0-15)
        duration_time = (datetime.datetime.now() - start_time).total_seconds()
        if duration_time > 0.02:
            click.secho(f"Sound command sending took a long time: {(duration_time):.3f}s", fg="red")
        # TODO: is it a good idea to have all this indexing? easy to screw up.

    def handle_midi_input(self):
        """Handle MIDI input messages that might come from mido"""
        if self.midi_in_port is None:
            return # fail early if MIDI not open.
        for message in self.midi_in_port.iter_pending():
            if message.type == "note_on":
                try:
                    index = self.config["midi"]["input"].index(["note_on", message.channel+1])
                    value = message.note / 127.0
                    self.callback(index,value)
                except ValueError:
                    pass

            if message.type == "control_change":
                try:
                    index = self.config["midi"]["input"].index(["control_change", message.channel+1, message.control])
                    value = message.value / 127.0
                    self.callback(index,value)
                except ValueError:
                    pass
