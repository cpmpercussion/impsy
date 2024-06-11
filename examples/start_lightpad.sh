#!/bin/bash
/Applications/Pd-0.53-2.app/Contents/Resources/bin/pd midi_controllers/roli-block-listener.pd &
poetry run ./start_imps.py run -D 4 -M s --log --verbose -O callresponse
# python predictive_music_model.py -d=4 --modelsize=s --log -v -c

