#!/bin/bash
/Applications/Pd-0.49-1.app/Contents/Resources/bin/pd midi_controllers/roli-block-listener.pd &
python predictive_music_model.py -d=4 --modelsize=s --log -v -c

