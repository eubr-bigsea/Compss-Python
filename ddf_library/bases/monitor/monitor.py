# -*- coding: utf-8 -*-

import sys
import io
import os
import time
import pickle
import pandas as pd
import multiprocessing
import logging

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
from flask import send_from_directory
from networkx.drawing.nx_agraph import graphviz_layout

log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

temporary_file = '/tmp/ddf_dash_object.pickle'

app = dash.Dash(__name__)
app.title = 'DDF Library Monitor'
app.css.config.serve_locally = True
app.scripts.config.serve_locally = True
app.layout = html.Div(
            className="row",
            children=[
                html.Link(rel='stylesheet', href='/static/stylesheet.css'),
                html.H1('DDF Monitor'),
                html.Div(id='live-update-elapsedtime'),
                html.Div(className="row",
                         children=[
                            html.Div(className="three columns",
                                     id='live-update-status',
                                     style={
                                            'height': '1000px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '2px',
                                            'textAlign': 'center'
                                            }),
                            dcc.Graph(className="nine columns",
                                      id='live-update-graph',
                                      style={
                                            'height': '1000px',
                                            'lineHeight': '60px',
                                            'borderWidth': '1px',
                                            'borderStyle': 'dashed',
                                            'borderRadius': '2px',
                                            'textAlign': 'center'
                                            },
                                      config={'scrollZoom': True,
                                              'modeBarButtonsToRemove':
                                                  ['sendDataToCloud',
                                                   'hoverClosestCartesian',
                                                   'hoverCompareCartesian',
                                                   'resetScale2d',
                                                   'toggleSpikelines'],
                                              'displaylogo': False}, ),
                            dcc.Interval(
                                    id='interval-component',
                                    interval=3 * 1000,  # in milliseconds
                                    n_intervals=0)
                            ])
            ])


@app.server.route('/static/<path:path>')
def serve_static(path):
    import ddf_library.bases.monitor
    folder = ddf_library.bases.monitor.__file__ \
        .replace('/__init__.py', '')
    static_folder = os.path.join(folder, 'static')

    return send_from_directory(static_folder, path)


def select_colors(status):
    if status == 'DELETED':
        color = 'lightgray'
    elif status == 'WAIT':
        color = 'yellow'
    elif status == 'PERSISTED':
        color = 'forestgreen'
    else:  # completed
        color = 'lightblue'
    return color


def gen_data(dag, catalog_tasks, msg_status, title):

    pos = graphviz_layout(dag, prog='dot')
    edge_x = []
    edge_y = []
    node_x = []
    node_y = []
    colors = []
    node_label = []
    information = []

    for node in pos:
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        status = catalog_tasks[node].get('status', 'WAIT')
        name = catalog_tasks[node]['name']
        node_label.append(name)

        color = select_colors(status)
        if name == 'init':
            color = 'white'
        colors.append(color)

        info = '<b>{} stage</b><br>' \
               'id: {}<br>' \
               'status: {}'.format(name, node, status)
        information.append(info)

    for edge in dag.edges():
        node0, node1 = edge
        x0, y0 = pos[node0]
        x1, y1 = pos[node1]
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

    data = {'node_x': node_x, 'node_y': node_y, 'edge_x': edge_x,
            'edge_y': edge_y, 'information': information, 'color': colors,
            'annotations': node_label, 'msg_status': msg_status,
            'title': title}

    with open(temporary_file, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


@app.callback(Output('live-update-elapsedtime', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_elapsed_time(n):
    style = {'padding': '5px', 'fontSize': '18px'}
    t = time.strftime('%b-%d-%Y at %H:%M', time.localtime())

    if os.path.exists(temporary_file):
        with open(temporary_file, 'rb') as handle:
            data = pickle.load(handle)

        title = data.get('title', '')
    else:
        title = '?'

    return [
        html.H2("Application id {}".format(title)),
        html.Span('Last update: {}'.format(t), style=style),
    ]


@app.callback(Output('live-update-status', 'children'),
              [Input('interval-component', 'n_intervals')])
def update_status(n):

    if os.path.exists(temporary_file):
        with open(temporary_file, 'rb') as handle:
            data = pickle.load(handle)

        table = pd.DataFrame(data.get('msg_status', [['-', '-']]),
                             columns=['Metric', 'Value'])
        msg = table.to_markdown()
        msg = "## Context Status: \n" + msg
    else:
        msg = "## Context Status: \n COMPSs Context is starting ..."

    return [
        dcc.Markdown(msg,
                     style={
                         'fontSize': '18px',
                         'text-align': 'center',
                         'margin-left': '10%',
                         'margin-top': '10%',
                     }),
    ]


@app.callback(Output('live-update-graph', 'figure'),
              [Input('interval-component', 'n_intervals')])
def update_graph_live(n):

    if os.path.exists(temporary_file):
        with open(temporary_file, 'rb') as handle:
            data = pickle.load(handle)
            annotations = []
            edge_x = data['edge_x']
            edge_y = data['edge_y']
            node_x = data['node_x']
            node_y = data['node_y']
            information = data['information']
            colors = data['color']
            node_names = data['annotations']

            edge_trace = go.Scatter(
                    x=edge_x, y=edge_y,
                    line=dict(width=2.5, color='#888'),
                    hoverinfo='none',
                    mode='lines')

            node_trace = go.Scatter(
                    x=node_x, y=node_y,
                    mode='markers',
                    textposition='bottom center',
                    hoverinfo='text',
                    text=information,
                    marker=dict(color=colors, size=20, line_width=2.5))

            for x, y, a in zip(node_x, node_y, node_names):
                annotations.append(
                        dict(x=x,
                             y=y,
                             text=a,  # node name that will be displayed
                             xanchor='left',
                             xshift=16,
                             font=dict(color='black', size=16),
                             showarrow=False, arrowhead=1, ax=-10, ay=-10)
                )

            fig = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                    title="DDF DAG",
                                    titlefont_size=26,
                                    showlegend=False,
                                    hovermode='closest',
                                    annotations=annotations,
                                    xaxis=dict(showgrid=False, zeroline=False,
                                               showticklabels=False),
                                    yaxis=dict(showgrid=False, zeroline=False,
                                               showticklabels=False)),

                            )

            return fig
    else:
        return go.Figure()


class Monitor(object):

    def __init__(self):
        """Start dash service."""
        self.p = None
        self.start()

    def start(self):
        """Start dash service."""

        sys.stdout = io.StringIO()
        port = 58227
        self.p = multiprocessing.Process(target=app.run_server,
                                         name="DDF_DASH",
                                         kwargs=dict(debug=False,
                                                     port=port)
                                         )
        self.p.start()
        sys.stdout = sys.__stdout__
        print(" * DDF Monitor is running on http://127.0.0.1:{}/".format(port))

    def stop(self):
        """
        It is necessary to kill the processes because of Dash live-update.
        :return:
        """
        self.p.terminate()
        import os
        os.system("kill -9 {}".format(str(self.p.pid)))
        return None
