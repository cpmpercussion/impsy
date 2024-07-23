from flask import Flask, render_template, request, send_file
import click
import os

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
MODEL_DIR = 'models'
CONFIGS_DIR = 'configs'
CONFIG_FILE = 'config.toml'
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 4000

ROUTE_NAMES = {
    'logs': 'Log Files',
    'edit_config': 'Edit Configuration',
}

def get_routes():
    page_routes = []
    for rule in app.url_map.iter_rules():
        if rule.endpoint != 'static' and 'GET' in rule.methods and '<' not in str(rule):
            page_routes.append({
                'endpoint': rule.endpoint,
                'route': str(rule)
            })
    return page_routes

@app.route('/')
def index():
    routes = get_routes()
    return render_template('index.html', routes=routes, route_names=ROUTE_NAMES)

@app.route('/logs')
def logs():
    log_files = os.listdir(LOGS_DIR)
    return render_template('logs.html', log_files=log_files)

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
        return render_template('edit_config.html', config_content=config_content)

def run_web_interface(host=DEFAULT_HOST, port=DEFAULT_PORT, debug=True):
    """Runs the Flask web interface."""
    click.secho(f'Starting web interface at http://{host}:{port}', fg='blue')
    click.secho(f'Log path: {LOGS_DIR}', fg='blue')
    app.run(host=host, port=port, debug=True)

@click.command(name="webui")
@click.option('--host', default=DEFAULT_HOST, help='The host to bind to.')
@click.option('--port', default=DEFAULT_PORT, help='The port to bind to.')
@click.option('--debug', is_flag=True, help='Enable debug mode.')
def webui(host, port, debug):
    """Run IMPSY Web UI giving access to files and other commands."""
    run_web_interface(host, port, debug)
