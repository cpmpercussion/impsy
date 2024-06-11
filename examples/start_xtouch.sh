#!/bin/bash
/Applications/Pd-0.53-2.app/Contents/Resources/bin/pd midi_controllers/xtouch-mini.pd &
poetry run ./start_imps.py run -D 9 -M s --log --verbose -O callresponse
