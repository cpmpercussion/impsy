from flask import Flask, render_template, request, send_file
import click
import os

app = Flask(__name__)

# Assume your MIDI files are stored in a 'log_files' directory
LOGS_DIR = 'logs'
MODEL_DIR = 'models'
CONFIGS_DIR = 'configs'
CONFIG_FILE = 'config.toml'
HOST = '0.0.0.0'
PORT = 6000

@app.route('/')
def index():
    log_files = os.listdir(LOGS_DIR)
    return render_template('templates/index.html', log_files=log_files)

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(LOGS_DIR, filename), as_attachment=True)

@app.route('/config', methods=['GET', 'POST'])
def edit_config():
    if request.method == 'POST':
        # Save the edited config file
        with open(CONFIG_FILE, 'w') as f:
            f.write(request.form['config_content'])
        return 'Config saved successfully'
    else:
        # Display the config file for editing
        with open(CONFIG_FILE, 'r') as f:
            config_content = f.read()
        return render_template('templates/edit_config.html', config_content=config_content)
    
@app.route("/hello")
def hello_world():
    return "<p>Hello, World!</p>"

def run_web_interface():
    """Runs the Flask web interface."""
    click.secho(f'Starting web interface at http://{HOST}:{PORT}', fg='blue')
    app.run(host=HOST, port=PORT, threaded=True)
