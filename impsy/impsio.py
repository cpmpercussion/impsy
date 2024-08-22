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
from pythonosc import dispatcher, osc_server, udp_client
from threading import Thread
from impsy.utils import get_midi_note_offs, output_values_to_midi_messages, match_midi_port_to_list, midi_message_to_index_value


class IOServer(abc.ABC):
    """Abstract class for music IO for IMPSY."""

    config: dict
    callback: Callable[[int, float], None]

    def __init__(
        self,
        config: dict,
        callback: Callable[[int, float], None],
        dense_callback: Callable[[List[int]], None],
    ) -> None:
        self.config = config  # the IMPSY config
        self.callback = callback  # a callback method to report incoming sparse data.(e.g., MIDI notes)
        self.dense_callback = dense_callback  # a callback for dense input data (e.g., lists of OSC arguments)

    @abc.abstractmethod
    def send(self, output_values) -> None:
        """Sends output values to relevant outputs."""
        pass

    @abc.abstractmethod
    def handle(self) -> None:
        """Handles input values (synchronously) if needed."""
        pass

    @abc.abstractmethod
    def connect(self) -> None:
        """Connect to inputs and outputs."""
        pass

    @abc.abstractmethod
    def disconnect(self) -> None:
        """Disconnect from inputs and outputs."""
        pass


class SerialServer(IOServer):
    """Handles standard serial communication for IMPSY. 
    Messages are encoded in CSV format with new lines at the end of each message."""


    def __init__(self, config: dict, callback: Callable[[int, float], None], dense_callback: Callable[[List[int]], None]) -> None:
        super().__init__(config, callback, dense_callback)
        self.serial_port = config["serial"]["port"]
        self.baudrate = config["serial"]["baudrate"] # 31250 midi, 
        self.serial = None
        self.buffer = ""

    
    def send(self, output_values) -> None:
        """Send values as a CSV line."""
        output_message = ','.join(f"{num:.4f}" for num in output_values) + '\n'
        # click.secho(f"Serial out: {output_message}")
        if self.serial is not None:
            self.serial.write(output_message.encode())
        else: 
            # try to reconnect -- may as well, alternative is just never working.
            self.connect()


    def handle(self) -> None:
        """read in the serial bytes and process lines into value lists for IMPSY"""
        # exist quickly if there is no serial connection.
        if self.serial is None:
            return
        
        # first read in the serial bytes in waiting
        while self.serial.in_waiting:
            try:
                self.buffer += self.serial.read(self.serial.in_waiting).decode()
            except Exception as e:
                click.secho(f"Serial: error decoding input {e}")
        
        # process lines into values.
        while '\n' in self.buffer:
            line, self.buffer = self.buffer.split('\n', 1)
            if line:
                try:
                    value_list = [float(x) for x in line.split(',')]
                    self.dense_callback(value_list) # callback with the value list.
                    # click.secho(f"Serial parsed: {value_list}", fg="green")
                except ValueError:
                    click.secho(f"Serial: Could not parse line: {line}", fg="red")


    def connect(self) -> None:
        """Tries to open a serial port for regular IO."""
        try:
            click.secho(f"Serial: Opening port {self.serial_port} at {self.baudrate} baud.", fg="yellow")
            self.serial = serial.Serial( self.serial_port, baudrate=self.baudrate, timeout=0)
        except:
            self.serial = None
            click.secho(f"Serial: Could not open {self.serial_port}.", fg="red")


    def disconnect(self) -> None:
        try:
            self.serial.close()
        except:
            pass
    

class SerialMIDIServer(IOServer):
    """Handles MIDI IO over serial."""

    def __init__(
        self,
        config: dict,
        callback: Callable[[int, float], None],
        dense_callback: Callable[[List[int]], None],
    ) -> None:
        super().__init__(config, callback, dense_callback)
        self.parser = mido.parser.Parser()
        self.serial_port = config["serial"]["port"]
        self.baudrate = 31250 # midi baudrate
        self.serial = None
        self.buffer = "" # used for storing serial data after reading
        self.last_midi_notes = {}  # dict to store last played notes via midi
        self.midi_output_mapping = self.config["serialmidi"]["output"]
        self.midi_input_mapping = self.config["serialmidi"]["input"]


    def send(self, output_values) -> None:
        """Sends sound commands via MIDI"""
        start_time = datetime.datetime.now()
        
        output_midi_messages = output_values_to_midi_messages(output_values, self.midi_output_mapping)
        for msg in output_midi_messages:
            # send note off if a previous note_on had been sent
            if msg.type == 'note_on' and msg.channel in self.last_midi_notes:
                note_off_msg = mido.Message("note_off", channel = msg.channel, note=self.last_midi_notes[msg.channel], velocity=0)
                self.send_midi_message(note_off_msg)
            self.send_midi_message(msg) # actually send the message.
            if msg.type == 'note_on':
                self.last_midi_notes[msg.channel] = msg.note # store last midi note if it was a note_on.

        duration_time = (datetime.datetime.now() - start_time).total_seconds()
        if duration_time > 0.02:
            click.secho(
                f"Sound command sending took a long time: {duration_time:.3f}s",
                fg="red",
            )

    def handle(self) -> None:
        """Read in some bytes from the serial port and try to handle any found MIDI messages."""
        if self.serial is None:
            return
        if self.serial.in_waiting >= 3:
            midi_bytes = self.serial.read(3)
            self.parser.feed(midi_bytes)
        message = self.parser.get_message()
        if message is None:
            return
        else:
            try:
                index, value = midi_message_to_index_value(message, self.midi_input_mapping)
                self.callback(index, value)
            except ValueError as e:
                # error when handling the MIDI message
                # click.secho(f"MIDISerial Handling failed for a message: {e}", fg="red")
                pass

    def connect(self) -> None:
        """Tries to open a serial port for regular IO."""
        try:
            click.secho(f"Serial: Opening port {self.serial_port} at {self.baudrate} baud. (MIDI mode)", fg="yellow")
            self.serial = serial.Serial( self.serial_port, baudrate=self.baudrate, timeout=0)
        except:
            self.serial = None
            click.secho(f"Serial: Could not open {self.serial_port}.", fg="red")


    def disconnect(self) -> None:
        try:
            self.serial.close()
        except:
            pass


    def send_midi_message(self, message):
        """Sends a mido MIDI message via the connected serial port."""
        if self.serial is not None:
            self.serial.write(message.bin())


    def send_midi_note_offs(self):
        """Sends note offs on any MIDI channels that have been used for notes."""
        note_off_messages = get_midi_note_offs(self.midi_output_mapping, self.last_midi_notes)
        for msg in note_off_messages:
            self.send_midi_message(msg)



class WebSocketServer(IOServer):
    """Handles Websocket Serving for IMPSY"""

    def __init__(self, config, callback, dense_callback) -> None:
        super().__init__(config, callback, dense_callback)
        self.ws_clients = set()  # storage for potential ws clients.
        self.ws_thread = None
        self.ws_server = None
        self.last_midi_notes = {}  # dict to store last played notes via midi
        self.midi_output_mapping = self.config["websocket"]["output"]
        self.midi_input_mapping = self.config["websocket"]["input"]

    def send(self, output_values) -> None:
        output_midi_messages = output_values_to_midi_messages(output_values, self.midi_output_mapping)
        for msg in output_midi_messages:
            # send note off if a previous note_on had been sent
            if msg.type == 'note_on' and msg.channel in self.last_midi_notes:
                note_off_msg = mido.Message("note_off", channel = msg.channel, note=self.last_midi_notes[msg.channel], velocity=0)
                self.websocket_send_midi(note_off_msg)
            self.websocket_send_midi(msg) # actually send the message.
            if msg.type == 'note_on':
                self.last_midi_notes[msg.channel] = msg.note # store last midi note if it was a note_on.

    def handle(self) -> None:
        return super().handle()
        # Don't need to handle anything as it works in a thread.

    def connect(self) -> None:
        click.secho("Preparing websocket thread.", fg="yellow")
        self.ws_thread = Thread(
            target=self.websocket_serve_loop, name="ws_receiver_thread", daemon=True
        )
        self.ws_thread.start()  # send it!

    def disconnect(self) -> None:
        if self.ws_server:
            self.ws_server.shutdown() # stops the server_forever loop on the server if it exists.
        try:
            self.ws_thread.join(timeout=1.0) # the shutdown poll is usually 0.5 seconds.
        except:
            pass
        if self.ws_server:
            self.ws_server.socket.close() 

    def websocket_send_midi(self, message):
        """Sends a mido MIDI message via websockets if available."""
        if message.type == "note_on":
            ws_msg = (
                f"/channel/{message.channel}/noteon/{message.note}/{message.velocity}"
            )
        elif message.type == "note_off":
            ws_msg = (
                f"/channel/{message.channel}/noteoff/{message.note}/{message.velocity}"
            )
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
        self.ws_clients.add(websocket)  # add websocket to the client list.
        # do the actual handling
        for message in websocket:
            click.secho(
                f"WS: {message}", fg="red"
            )  # TODO: fine for debug, but should be removed really.
            m = message.split("/")[1:]
            msg_type = m[2]
            chan = int(m[1])  # TODO: should this be chan+1 or -1 or something.
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
                    index = self.config["midi"]["input"].index(
                        ["control_change", chan, note]
                    )
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
        hostname = self.config["websocket"]["server_ip"]
        port = self.config["websocket"]["server_port"]
        with serve(self.websocket_handler, hostname, port) as server:
            self.ws_server = server
            server.serve_forever()


class OSCServer(IOServer):
    """Handles OSC IO for IMPSY."""

    # [osc]
    # server_ip = "localhost" # Address of IMPSY
    # server_port = 5000 # Port IMPSY listens on
    # client_ip = "localhost" # Address of the output device
    # client_port = 5002 # Port of the output device

    # Details for OSC output
    INPUT_MESSAGE_ADDRESS = "/interface"
    OUTPUT_MESSAGE_ADDRESS = "/impsy"
    TEMPERATURE_MESSAGE_ADDRESS = "/temperature"
    TIMESCALE_MESSAGE_ADDRESS = "/timescale"

    def __init__(self, config, callback, dense_callback) -> None:
        super().__init__(config, callback, dense_callback)
        # Set up OSC client and server
        self.dimension = config["model"][
            "dimension"
        ]  # retrieve dimension from the config file.
        self.verbose = config["verbose"]
        self.osc_client = udp_client.SimpleUDPClient(
            config["osc"]["client_ip"], config["osc"]["client_port"]
        )
        self.dispatcher = dispatcher.Dispatcher()
        self.dispatcher.map(
            OSCServer.INPUT_MESSAGE_ADDRESS, self.handle_interface_message
        )
        self.dispatcher.map(
            OSCServer.TEMPERATURE_MESSAGE_ADDRESS, self.handle_temperature_message
        )
        self.dispatcher.map(
            OSCServer.TIMESCALE_MESSAGE_ADDRESS, self.handle_timescale_message
        )
        self.server = osc_server.ThreadingOSCUDPServer(
            (config["osc"]["server_ip"], config["osc"]["server_port"]), self.dispatcher
        )

    def handle_interface_message(self, address: str, *osc_arguments) -> None:
        self.dense_callback([*osc_arguments])

    def handle_temperature_message(self, address: str, *osc_arguments) -> None:
        """Handler for temperature messages from the interface: format is ff [sigma temp, pi temp]"""
        new_sigma_temp = osc_arguments[0]
        new_pi_temp = osc_arguments[1]
        if self.verbose:
            click.secho(
                f"Temperature -- Sigma: {new_sigma_temp}, Pi: {new_pi_temp}", fg="blue"
            )
        # TODO, set the network temperature somehow.
        # net.sigma_temp = new_sigma_temp
        # net.pi_temp = new_pi_temp

    def handle_timescale_message(self, address: str, *osc_arguments) -> None:
        """Handler for timescale messages: format is f [timescale]"""
        new_timescale = osc_arguments[0]
        if self.verbose:
            click.secho(f"Timescale: {new_timescale}", fg="blue")
        # TODO: do something with this information...

    def connect(self) -> None:
        click.secho("Preparing OSC server thread.", fg="yellow")
        self.server_thread = Thread(
            target=self.server.serve_forever, name="osc_server_thread", daemon=True
        )
        self.server_thread.start()

    def disconnect(self) -> None:
        if self.server:
            self.server.shutdown()
        try:
            self.server_thread.join(timeout=1.0)
        except:
            pass
        if self.server:
            self.server.socket.close()

    def send(self, output_values) -> None:
        try:
            self.osc_client.send_message(OSCServer.OUTPUT_MESSAGE_ADDRESS, output_values)
        except Exception as e:
            click.secho(f"OSC sending failed: {e}", fg="red")

    def handle(self) -> None:
        return super().handle()


class MIDIServer(IOServer):
    """Handles MIDI IO for IMPSY."""


    def __init__(self, config, callback, dense_callback) -> None:
        super().__init__(config, callback, dense_callback)
        self.dimension = self.config["model"][
            "dimension"
        ]  # retrieve dimension from the config file.
        self.verbose = self.config["verbose"]
        self.last_midi_notes = {}  # dict to store last played notes via midi
        self.midi_output_mapping = self.config["midi"]["output"]
        self.midi_input_mapping = self.config["midi"]["input"]
        # self.websocket_send_midi = None  # TODO implement some kind generic MIDI callback for other output channels.


    def send(self, output_values) -> None:
        """Sends sound commands via MIDI"""
        assert (
            len(output_values) + 1 == self.dimension
        ), "Dimension not same as prediction size."  # Todo more useful error.
        
        output_midi_messages = output_values_to_midi_messages(output_values, self.midi_output_mapping)
        for msg in output_midi_messages:
            # send note off if a previous note_on had been sent
            if msg.type == 'note_on' and msg.channel in self.last_midi_notes:
                note_off_msg = mido.Message("note_off", channel = msg.channel, note=self.last_midi_notes[msg.channel], velocity=0)
                self.send_midi_message(note_off_msg)
            # actually send the message.
            self.send_midi_message(msg)
            # store last midi note if it was a note_on.
            if msg.type == 'note_on':
                self.last_midi_notes[msg.channel] = msg.note
    

    def handle(self) -> None:
        """Handle MIDI input messages that might come from mido"""
        if self.midi_in_port is None:
            return  # fail early if MIDI not open.
        for message in self.midi_in_port.iter_pending():
            try:
                index, value = midi_message_to_index_value(message, self.midi_input_mapping)
                self.callback(index, value)
            except ValueError as e:
                # error when handling the MIDI message
                # click.secho(f"MIDI Handling failed for a message: {e}", fg="red")
                pass


    def connect(self) -> None:
        """Opens MIDI Ports"""
        click.secho("Opening MIDI port for input/output.", fg="yellow")
        potential_midi_inputs = []
        potential_midi_outputs = []
        try:
            potential_midi_inputs = mido.get_input_names()
            desired_input_port = match_midi_port_to_list(
                self.config["midi"]["in_device"], potential_midi_inputs
            )
            self.midi_in_port = mido.open_input(desired_input_port)
            click.secho(f"MIDI: in port is: {self.midi_in_port.name}", fg="green")
        except:
            self.midi_in_port = None
            click.secho("Could not open MIDI input.", fg="red")
            click.secho(f"MIDI Inputs: {potential_midi_inputs}", fg="blue")
        try:
            potential_midi_outputs = mido.get_output_names()
            desired_output_port = match_midi_port_to_list(
                self.config["midi"]["out_device"], potential_midi_outputs
            )
            self.midi_out_port = mido.open_output(desired_output_port)
            click.secho(f"MIDI: out port is: {self.midi_out_port.name}", fg="green")
        except:
            self.midi_out_port = None
            click.secho("Could not open MIDI output.", fg="red")
            click.secho(f"MIDI Outputs: {potential_midi_outputs}", fg="blue")


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


    def send_midi_message(self, message):
        """Send a MIDI message across all required outputs"""
        if self.midi_out_port is not None:
            self.midi_out_port.send(message)
        # if self.websocket_send_midi is not None:
        #     self.websocket_send_midi(message)


    def send_midi_note_offs(self):
        """Sends note offs on any MIDI channels that have been used for notes."""
        note_off_messages = get_midi_note_offs(self.midi_output_mapping, self.last_midi_notes)
        for msg in note_off_messages:
            self.send_midi_message(msg)
