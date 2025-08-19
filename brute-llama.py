import dash
from dash import dcc, html, no_update, ctx
from dash.dependencies import Input, Output, State, ALL
import plotly.express as px
import pandas as pd
import os
import glob
import shlex
import subprocess
import threading
import time
import requests
from datetime import datetime
import itertools
import uuid
import dash_bootstrap_components as dbc
import yaml

# --- Global Configuration & Thread Control ---
CONFIG_DIR = "configs"
DATA_DIR = "data"
LOG_FILE = os.path.join(DATA_DIR, "test_run.log")
RESULTS_FILE = os.path.join(DATA_DIR, "results.csv")
ABORT_EVENT = threading.Event()
CURRENT_PROCESS = None
os.makedirs(CONFIG_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)


# ==============================================================================
# PROCESS AND DATA LOGIC (Stable)
# ==============================================================================
def start_instance(command: str) -> subprocess.Popen:
    args = shlex.split(command)
    return subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )


def stop_instance(process: subprocess.Popen):
    global CURRENT_PROCESS
    if process and process.poll() is None:
        try:
            process.terminate()
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
    CURRENT_PROCESS = None


def wait_for_api(base_url: str, timeout: int = 300) -> bool:
    health_url = f"{base_url}/health"
    start_time = time.time()
    append_to_log(f"Waiting for health check at '{health_url}'...")
    while time.time() - start_time < timeout:
        if ABORT_EVENT.is_set():
            return False
        try:
            response = requests.get(health_url, timeout=5)
            if response.status_code == 200 and response.json() == {"status": "ok"}:
                append_to_log("Health check successful (status: ok). Server is ready.")
                return True
            elif response.status_code == 503:
                append_to_log(
                    "Server reports 503 (Service Unavailable), model is likely loading. Waiting..."
                )
        except requests.RequestException:
            append_to_log("Server not yet reachable. Waiting...")
        time.sleep(3)
    append_to_log(f"ERROR: Timeout after {timeout}s. Server did not become ready.")
    return False


def get_measurement(command: str, is_debug: bool) -> float | None:
    final_command = command
    if is_debug:
        if "|" in command:
            final_command = command.split("|", 1)[0].strip()
        append_to_log(f"DEBUG MODE: Executing only command part: '{final_command}'")
    try:
        result = subprocess.run(
            final_command,
            shell=True,
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        output = result.stdout.strip()
        if is_debug:
            append_to_log(
                f"DEBUG MODE: Unfiltered server response:\n---\n{output}\n---"
            )
            return -1.0
        if not output:
            append_to_log(
                "ERROR: Measurement command returned an empty string. Check your jq filter."
            )
            return None
        return float(output)
    except (subprocess.CalledProcessError, ValueError, subprocess.TimeoutExpired) as e:
        stderr = e.stderr if hasattr(e, "stderr") else ""
        stdout = e.stdout if hasattr(e, "stdout") else ""
        append_to_log(
            f"ERROR during measurement: {e}\nSTDOUT: {stdout}\nSTDERR: {stderr}"
        )
        return None


def log_measurement_to_csv(params_dict, measurement):
    data_row = params_dict.copy()
    data_row["timestamp"] = datetime.now()
    data_row["measurement"] = measurement
    new_data = pd.DataFrame([data_row])
    try:
        df = pd.read_csv(RESULTS_FILE)
        df = pd.concat([df, new_data], ignore_index=True)
    except (FileNotFoundError, pd.errors.EmptyDataError):
        df = new_data
    df.to_csv(RESULTS_FILE, index=False)


def append_to_log(message: str):
    with open(LOG_FILE, "a") as f:
        f.write(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")


def run_test_sequence(
    server_params,
    measure_params,
    server_template,
    measure_template,
    server_url,
    is_debug,
):
    global CURRENT_PROCESS
    try:
        append_to_log(">>> Test run started.")
        if not all([server_template, measure_template, server_url]):
            append_to_log("FATAL ERROR: One or more template/URL fields are empty.")
            return
        if not server_url.startswith(("http://", "https://")):
            server_url = "http://" + server_url
        s_names = [p["name"] for p in server_params]
        s_vals = [p["values"].split(";") for p in server_params]
        m_names = [p["name"] for p in measure_params]
        m_vals = [p["values"].split(";") for p in measure_params]
        s_prod = list(itertools.product(*s_vals)) if s_vals else [()]
        m_prod = list(itertools.product(*m_vals)) if m_vals else [()]
        for s_combo in s_prod:
            if ABORT_EVENT.is_set():
                break
            current_s_params = dict(zip(s_names, s_combo))
            process = None
            try:
                start_cmd = server_template
                for name, value in current_s_params.items():
                    start_cmd = start_cmd.replace(f"{{{{{name}}}}}", value.strip())
                append_to_log(
                    f"---[Server Loop] Starting instance with {current_s_params} ---"
                )
                append_to_log(f"Command: {start_cmd}")
                process = start_instance(start_cmd)
                CURRENT_PROCESS = process
                if wait_for_api(server_url):
                    for m_combo in m_prod:
                        if ABORT_EVENT.is_set():
                            break
                        current_m_params = dict(zip(m_names, m_combo))
                        append_to_log(
                            f"  --[Measure Loop] Testing with {current_m_params} --"
                        )
                        measure_cmd = measure_template
                        all_params = {**current_s_params, **current_m_params}
                        for name, value in all_params.items():
                            measure_cmd = measure_cmd.replace(
                                f"{{{{{name}}}}}", value.strip()
                            )
                        time.sleep(1)
                        measurement = get_measurement(measure_cmd, is_debug)
                        if measurement is not None:
                            if not is_debug:
                                log_measurement_to_csv(all_params, measurement)
                            append_to_log(f"  Measurement result: {measurement}")
                else:
                    append_to_log(f"Server start failed.")
            finally:
                append_to_log(f"---[Server Loop] Stopping instance ---")
                stop_instance(process)
                append_to_log("-" * 50)
                time.sleep(2)
    except Exception as e:
        append_to_log(f"FATAL ERROR in test run wrapper: {e}")
    finally:
        append_to_log(">>> Test run finished.")
        ABORT_EVENT.clear()


# ==============================================================================
# DASH WEB UI
# ==============================================================================
app = dash.Dash(
    __name__,
    suppress_callback_exceptions=True,
    external_stylesheets=[dbc.themes.DARKLY],
)


def create_param_row_layout(p_type, p_id, p_name="", p_values=""):
    return dbc.Row(
        [
            dbc.Col(
                dbc.Input(
                    value=p_name,
                    id={"type": f"{p_type}-param-name", "index": p_id},
                    placeholder="Parameter Name",
                ),
                width=3,
            ),
            dbc.Col(
                dbc.Input(
                    value=p_values,
                    id={"type": f"{p_type}-param-values", "index": p_id},
                    placeholder="Values (semicolon-separated)",
                ),
                width=8,
            ),
            dbc.Col(
                dbc.Button(
                    "-",
                    id={"type": f"remove-{p_type}-param", "index": p_id},
                    n_clicks=0,
                    color="danger",
                ),
                width=1,
            ),
        ],
        id={"type": f"{p_type}-param-row", "index": p_id},
        className="mb-2",
    )


def get_saved_configs():
    files = glob.glob(os.path.join(CONFIG_DIR, "*.yaml"))
    return sorted([os.path.splitext(os.path.basename(f))[0] for f in files])


"""
navbar = dbc.NavbarSimple(
    brand="brute-llama",
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4",
)
"""

navbar = dbc.NavbarSimple(
    brand=html.Div(
        [
            html.Img(
                src="/assets/logo.png",
                height="30px",
                style={"marginRight": "10px"},
            ),
            "brute-llama",
        ],
        style={"display": "flex", "alignItems": "center"},
    ),
    brand_href="#",
    color="primary",
    dark=True,
    className="mb-4",
)

config_management = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Configuration Management", className="card-title"),
            dbc.Label("Load saved configuration:"),
            dcc.Dropdown(
                id="config-dropdown",
                options=get_saved_configs(),
                placeholder="Select a configuration...",
            ),
            html.Hr(),
            dbc.Label("Save current configuration as:"),
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Input(
                            id="config-name-input", placeholder="Configuration name"
                        ),
                        width=9,
                    ),
                    dbc.Col(
                        dbc.Button(
                            "Save",
                            id="save-config-button",
                            color="primary",
                            className="w-100",
                        ),
                        width=3,
                    ),
                ]
            ),
            dbc.Alert(
                id="save-feedback", is_open=False, duration=4000, className="mt-3"
            ),
        ]
    ),
    className="mb-3",
)

settings_accordion = dbc.Accordion(
    [
        dbc.AccordionItem(
            [
                config_management,
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("Server Parameters", className="card-title"),
                            dbc.Label("Server Start Command Template:"),
                            dbc.Input(
                                id="server-command-template",
                                type="text",
                                placeholder="./server --model {{model_name}}",
                                className="mb-2",
                            ),
                            html.Div(id="server-params-container"),
                            dbc.Button(
                                "+ Add Server Parameter",
                                id="add-server-param",
                                n_clicks=0,
                                color="secondary",
                                className="mt-2",
                            ),
                        ]
                    ),
                    className="mb-3",
                ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("Measurement Parameters", className="card-title"),
                            dbc.Label("Measurement Command Template:"),
                            dbc.Textarea(
                                id="measure-command-template",
                                placeholder="curl ... | jq .timings.predicted_per_second",
                                style={"height": "100px"},
                                className="mb-2",
                            ),
                            html.Div(id="measure-params-container"),
                            dbc.Button(
                                "+ Add Measurement Parameter",
                                id="add-measure-param",
                                n_clicks=0,
                                color="secondary",
                                className="mt-2",
                            ),
                        ]
                    ),
                    className="mb-3",
                ),
                dbc.Card(
                    dbc.CardBody(
                        [
                            html.H4("General Settings", className="card-title"),
                            dbc.Label("Server URL (for Health Check):"),
                            dbc.Input(
                                id="server-url",
                                type="text",
                                placeholder="127.0.0.1:8080",
                                className="mb-2",
                            ),
                            dbc.Checklist(
                                id="debug-mode-checkbox",
                                options=[{"label": "Debug Mode", "value": "DEBUG"}],
                                value=[],
                                switch=True,
                            ),
                            html.Div(
                                id="button-container", className="mt-3 d-grid gap-2"
                            ),
                        ]
                    )
                ),
            ],
            title="Configuration & Control",
        )
    ],
    start_collapsed=False,
)

dashboard = dbc.Card(
    dbc.CardBody(
        [
            html.H4("Dashboard", className="card-title"),
            dbc.Row(
                [
                    dbc.Col(
                        [dbc.Label("X-Axis:"), dcc.Dropdown(id="xaxis-dropdown")],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Curve Color (Group by):"),
                            dcc.Dropdown(id="color-dropdown"),
                        ],
                        width=4,
                    ),
                    dbc.Col(
                        [
                            dbc.Label("Facets (Grid Columns):"),
                            dcc.Dropdown(id="facet-dropdown"),
                        ],
                        width=4,
                    ),
                ],
                className="mb-3",
            ),
            html.H5("Live Log", className="mt-3"),
            dbc.Spinner(
                html.Pre(
                    id="log-output",
                    style={
                        "height": "300px",
                        "overflowY": "scroll",
                        "backgroundColor": "#1E1E1E",
                        "border": "1px solid #444",
                        "padding": "10px",
                    },
                )
            ),
            html.H5("Performance Plot", className="mt-3"),
            dbc.Spinner(dcc.Graph(id="performance-graph")),
        ]
    )
)

app.layout = dbc.Container(
    [
        dcc.Store(id="server-params-store", data=[]),
        dcc.Store(id="measure-params-store", data=[]),
        dcc.Store(id="is-running-state", data=False),
        dcc.Interval(
            id="interval-component", interval=2000, n_intervals=0, disabled=True
        ),
        navbar,
        settings_accordion,
        dashboard,
    ],
    fluid=True,
)

# --- Callbacks ---
for p_type in ["server", "measure"]:

    @app.callback(
        Output(f"{p_type}-params-container", "children"),
        Input(f"{p_type}-params-store", "data"),
    )
    def render_params(data, p_type=p_type):
        return [
            create_param_row_layout(p_type, p["id"], p["name"], p["values"])
            for p in data
        ]

    @app.callback(
        Output(f"{p_type}-params-store", "data", allow_duplicate=True),
        Input(f"add-{p_type}-param", "n_clicks"),
        State(f"{p_type}-params-store", "data"),
        prevent_initial_call=True,
    )
    def add_param(n_clicks, data):
        data.append({"id": str(uuid.uuid4()), "name": "", "values": ""})
        return data

    @app.callback(
        Output(f"{p_type}-params-store", "data", allow_duplicate=True),
        Input({"type": f"remove-{p_type}-param", "index": ALL}, "n_clicks"),
        State(f"{p_type}-params-store", "data"),
        prevent_initial_call=True,
    )
    def remove_param(n_clicks, data):
        if not any(n > 0 for n in n_clicks if n):
            return no_update
        clicked_id = ctx.triggered_id["index"]
        return [p for p in data if p["id"] != clicked_id]

    @app.callback(
        Output(f"{p_type}-params-store", "data", allow_duplicate=True),
        [
            Input({"type": f"{p_type}-param-name", "index": ALL}, "value"),
            Input({"type": f"{p_type}-param-values", "index": ALL}, "value"),
        ],
        State(f"{p_type}-params-store", "data"),
        prevent_initial_call=True,
    )
    def sync_store(names, values, data):
        data_dict = {p["id"]: p for p in data}
        for i, name in enumerate(names):
            p_id = ctx.inputs_list[0][i]["id"]["index"]
            if p_id in data_dict:
                data_dict[p_id]["name"] = name
        for i, value in enumerate(values):
            p_id = ctx.inputs_list[1][i]["id"]["index"]
            if p_id in data_dict:
                data_dict[p_id]["values"] = value
        return list(data_dict.values())


@app.callback(Output("button-container", "children"), Input("is-running-state", "data"))
def update_buttons(is_running):
    if is_running:
        return dbc.Button(
            "CANCEL TEST RUN", id="abort-button", color="danger", size="lg"
        )
    return dbc.Button("START TEST RUN", id="start-button", color="success", size="lg")


@app.callback(
    [
        Output("is-running-state", "data", allow_duplicate=True),
        Output("interval-component", "disabled", allow_duplicate=True),
    ],
    Input("start-button", "n_clicks"),
    [
        State("server-params-store", "data"),
        State("measure-params-store", "data"),
        State("server-command-template", "value"),
        State("measure-command-template", "value"),
        State("server-url", "value"),
        State("debug-mode-checkbox", "value"),
    ],
    prevent_initial_call=True,
)
def handle_start(n_clicks, s_params, m_params, s_tpl, m_tpl, url, debug_val):
    if not n_clicks:
        return no_update, no_update
    s_params_filtered = [p for p in s_params if p.get("name") and p.get("values")]
    m_params_filtered = [p for p in m_params if p.get("name") and p.get("values")]
    is_debug = "DEBUG" in debug_val
    with open(LOG_FILE, "w") as f:
        f.write("")
    with open(RESULTS_FILE, "w") as f:
        f.write("")
    ABORT_EVENT.clear()
    thread = threading.Thread(
        target=run_test_sequence,
        name="TestSequenceThread",
        args=(s_params_filtered, m_params_filtered, s_tpl, m_tpl, url, is_debug),
    )
    thread.start()
    return True, False


@app.callback(
    Output("is-running-state", "data", allow_duplicate=True),
    Input("abort-button", "n_clicks"),
    prevent_initial_call=True,
)
def handle_abort(n_clicks):
    if not n_clicks:
        return no_update
    append_to_log("!!! ABORT BUTTON PRESSED !!!")
    ABORT_EVENT.set()
    stop_instance(CURRENT_PROCESS)
    return True


@app.callback(
    Output("log-output", "children"), Input("interval-component", "n_intervals")
)
def update_logs(n):
    try:
        with open(LOG_FILE, "r") as f:
            return f.read()
    except FileNotFoundError:
        return "Waiting for test run to start..."


@app.callback(
    [
        Output("xaxis-dropdown", "options"),
        Output("color-dropdown", "options"),
        Output("facet-dropdown", "options"),
    ],
    [Input("server-params-store", "data"), Input("measure-params-store", "data")],
)
def update_dropdown_options(server_params, measure_params):
    s_names = [p["name"] for p in server_params if p.get("name")]
    m_names = [p["name"] for p in measure_params if p.get("name")]
    all_names = sorted(list(set(s_names + m_names)))
    options = [{"label": name, "value": name} for name in all_names]
    return options, options, options


@app.callback(
    Output("performance-graph", "figure"),
    [
        Input("xaxis-dropdown", "value"),
        Input("color-dropdown", "value"),
        Input("facet-dropdown", "value"),
    ],
    Input("interval-component", "n_intervals"),
)
def update_graph(xaxis, color, facet, n):
    template = "plotly_dark"
    triggered_id = ctx.triggered_id
    if triggered_id != "interval-component" and n > 0:
        return no_update
    if not xaxis:
        return px.line(title="Select an X-axis to view plot", template=template)
    try:
        df = pd.read_csv(RESULTS_FILE)
        for col in df.columns:
            if col not in ["timestamp"]:
                df[col] = pd.to_numeric(df[col], errors="ignore")

        fig = px.line(
            df,
            x=xaxis,
            y="measurement",
            color=color,
            facet_col=facet,
            markers=True,
            title="Performance-Analysis",
            labels={"measurement": "Measurement"},
            template=template,
        )
        # NEU: Diese Zeile aktiviert die sanften Übergänge
        # fig.update_layout(transition_duration=500)
        return fig

        # return px.line(df, x=xaxis, y='measurement', color=color, facet_col=facet,
        #               markers=True, title="Performance Analysis",
        #               labels={"measurement": "Measurement"}, template=template)

    except (FileNotFoundError, pd.errors.EmptyDataError, KeyError):
        return px.line(title="Waiting for data...", template=template)


@app.callback(
    [
        Output("is-running-state", "data", allow_duplicate=True),
        Output("interval-component", "disabled"),
    ],
    Input("interval-component", "n_intervals"),
    State("is-running-state", "data"),
    prevent_initial_call=True,
)
def check_thread_status_and_disable_interval(n, is_running):
    if is_running and not any(
        "TestSequenceThread" in t.name for t in threading.enumerate()
    ):
        return False, True
    return no_update, no_update


@app.callback(
    [
        Output("save-feedback", "children"),
        Output("save-feedback", "is_open"),
        Output("config-dropdown", "options"),
    ],
    Input("save-config-button", "n_clicks"),
    [
        State("config-name-input", "value"),
        State("server-command-template", "value"),
        State("measure-command-template", "value"),
        State("server-url", "value"),
        State("debug-mode-checkbox", "value"),
        State("server-params-store", "data"),
        State("measure-params-store", "data"),
    ],
    prevent_initial_call=True,
)
def save_config(n_clicks, name, s_tpl, m_tpl, url, debug, s_params, m_params):
    if not name:
        return "Please enter a name for the configuration.", True, no_update
    config_data = {
        "server_template": s_tpl,
        "measure_template": m_tpl,
        "server_url": url,
        "debug_mode": debug,
        "server_params": s_params,
        "measure_params": m_params,
    }
    filepath = os.path.join(CONFIG_DIR, f"{name}.yaml")
    with open(filepath, "w") as f:
        yaml.dump(config_data, f, sort_keys=False)
    new_options = get_saved_configs()
    return f"Configuration '{name}' saved successfully!", True, new_options


@app.callback(
    [
        Output("server-command-template", "value"),
        Output("measure-command-template", "value"),
        Output("server-url", "value"),
        Output("debug-mode-checkbox", "value"),
        Output("server-params-store", "data"),
        Output("measure-params-store", "data", allow_duplicate=True),
        Output("config-name-input", "value"),
    ],
    Input("config-dropdown", "value"),
    prevent_initial_call=True,
)
def load_config(name):
    if not name:
        return no_update
    filepath = os.path.join(CONFIG_DIR, f"{name}.yaml")
    if os.path.exists(filepath):
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        return (
            data.get("server_template", ""),
            data.get("measure_template", ""),
            data.get("server_url", ""),
            data.get("debug_mode", []),
            data.get("server_params", []),
            data.get("measure_params", []),
            name,
        )
    return no_update


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=9111)
