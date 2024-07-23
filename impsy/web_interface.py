from flask import Flask, render_template, request, send_file, redirect, url_for
from werkzeug.utils import secure_filename
import click
import psutil
import shutil
import platform
import os
import tomllib

app = Flask(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
LOGS_DIR = os.path.join(PROJECT_ROOT, 'logs')
MODEL_DIR = os.path.join(PROJECT_ROOT, 'models')
DATASET_DIR = os.path.join(PROJECT_ROOT, 'datasets')
CONFIGS_DIR = os.path.join(PROJECT_ROOT, 'configs')
CONFIG_FILE = 'config.toml'
DEFAULT_HOST = '127.0.0.1'
DEFAULT_PORT = 4000

ROUTE_NAMES = {
    'logs': 'Log Files',
    'edit_config': 'Edit Configuration',
    'models': 'Model Files',
    'datasets': 'Dataset Files',
}

@app.template_filter("startswith")
def test_startswith(s, start):
    return s.startswith(start)

def get_hardware_info():
    try:
        cpu_info = platform.machine()
        cpu_cores = psutil.cpu_count(logical=False)
        disk_usage = shutil.disk_usage('/')
        memory = psutil.virtual_memory()
        ram_total = memory.total / (1024 ** 3)
        ram_used = (memory.total - memory.available) / (1024 ** 3)
        disk = disk_usage.free / (1024 ** 3)
        disk_percent = 100 * disk_usage.used / disk_usage.total
        os_info = f"{platform.system()} {platform.release()}"
        print(psutil.disk_usage('/'))
        return {
            "CPU": cpu_info,
            "CPU Cores": cpu_cores,
            "RAM": f"{ram_used:.2f}/{ram_total:.2f} GB ({memory.percent}% used)",
            "Disk Space Free": f"{disk:.2f} GB ({disk_percent:.2f}% used)",
            "OS": os_info
        }
    except Exception as e:
        return {"Error": str(e)}
    
def get_software_info():
    with open("pyproject.toml", "rb") as f:
        pyproject_data = tomllib.load(f)
    return {
        "Project": pyproject_data["tool"]["poetry"].get("name"),
        "Version": pyproject_data["tool"]["poetry"].get("version"),
        "Description": pyproject_data["tool"]["poetry"].get("description"),
        "Authors": pyproject_data["tool"]["poetry"].get("authors"),
        "Homepage": pyproject_data["tool"]["poetry"].get("homepage"),
        "Repository": pyproject_data["tool"]["poetry"].get("repository"),
    }


def allowed_model_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'keras', 'h5', 'tflite'} 

def allowed_log_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'log'} 

def allowed_dataset_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'npz'} 


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
    hardware_info = get_hardware_info()
    software_info = get_software_info()
    return render_template('index.html', routes=routes, route_names=ROUTE_NAMES, hardware_info=hardware_info, software_info=software_info)

@app.route('/logs')
def logs():
    log_files = [f for f in os.listdir(LOGS_DIR) if allowed_log_file(f)]
    return render_template('logs.html', log_files=log_files)

@app.route('/datasets')
def datasets():
    dataset_files = [f for f in os.listdir(DATASET_DIR) if allowed_dataset_file(f)]
    print(dataset_files)
    return render_template('datasets.html', dataset_files=dataset_files)

@app.route('/models', methods=['GET', 'POST'])
def models():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_model_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(MODEL_DIR, filename))
            return redirect(url_for('models'))
    
    model_files = [f for f in os.listdir(MODEL_DIR) if allowed_model_file(f)]
    return render_template('models.html', model_files=model_files)

@app.route('/download_log/<filename>')
def download_log(filename):
    return send_file(os.path.join(LOGS_DIR, filename), as_attachment=True)

@app.route('/download_model/<filename>')
def download_model(filename):
    return send_file(os.path.join(MODEL_DIR, filename), as_attachment=True)

@app.route('/download_dataset/<filename>')
def download_dataset(filename):
    return send_file(os.path.join(DATASET_DIR, filename), as_attachment=True)

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
