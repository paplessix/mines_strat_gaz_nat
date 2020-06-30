from datetime import datetime as dt
import dash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
from plotly.graph_objs import *
import numpy as np
import pandas as pd
import random as rd
from strat_gaz.storage_optimisation.matrices import Matrices


external_stylesheets = []

app = dash.Dash(__name__, suppress_callback_exceptions=True, external_stylesheets=external_stylesheets)

spot_history_df = pd.read_csv('strat_gaz/scrap/last_save/spot_€_MWh_PEG.csv')
forward_history_df = pd.read_csv('strat_gaz/scrap/last_save/forward_€_MWh_PEG.csv')
scenario_df = pd.read_csv('strat_gaz/Data/Diffusion/Diffusion_model_dynamic_forward_1000.csv')
strat_df = pd.read_csv('strat_gaz/storage_optimisation/results/SedN_100_40_o.csv')

#https://stackoverflow.com/questions/45577255/plotly-plot-multiple-figures-as-subplots

header = html.Div(
    children = 'Mines strat gaz nat',
    className = 'row pretty-border'
)

body_layout = html.Div([

    # Composant données historiques
    html.Div([
        dbc.Tabs(
            [
                dbc.Tab(label="Spot", tab_id="tab-1"),
                dbc.Tab(label="Forward", tab_id="tab-2"),
            ],
            id="tabs",
            active_tab="tab-1",
        ),
        html.Div(id="content"),
    ], className = 'col-sm pretty-border'),


    #Composant présentation scénarii
    html.Div([

        dcc.Graph(
            id='scenario',
            config = {'displayModeBar': False}
        ),

        #Collapse pour montrer/cacher les infos de volatilité
        dbc.Button(
            "Scénario aléatoire",
            id="random-button",
            className="mb-3",
            color="primary"
        ),
        dbc.Button(
            "Détails",
            id="collapse-button",
            className="mb-3",
            color="light"
        ),
        dbc.Collapse(
            dbc.Table(
                html.Tbody([
                    html.Tr([html.Th(" "), html.Th("Hiver"), html.Th("Ete")]),
                    html.Tr([html.Th("Semaine"), html.Th("26%"), html.Th("35%")]),
                    html.Tr([html.Th("Week-end"), html.Th("13%"), html.Th("17%")])
                ]),
                bordered = True
            ),
            id = 'collapse'
        )
    ], className = 'col-sm scenar pretty-border text-center')

], className = 'row')

app.layout = html.Div(
    children = [header, body_layout],
    className = 'container-fluid'
)


################# Callbacks ##################

#Callback qui gère les onglets de données historiques
@app.callback(Output("content", "children"), [Input("tabs", "active_tab")])
def switch_tab(at):

    spot_content = html.Div([
        dcc.Graph(
            id='historical-spot',
            config = {'displayModeBar': False}
        ),

        dcc.Slider(
            id='year-slider-spot',
            min=3,
            max=5,
            value=3,
            marks = {3: 'Mar', 4: 'Apr', 5: 'May'},
            step=None
        )
    ], className = 'histo')

    forward_content = html.Div([
        dcc.Graph(
            id='historical-forward',
            config = {'displayModeBar': False}
        ),

        dcc.DatePickerSingle(
            id='date-picker-forward',
            min_date_allowed=dt(2020, 2, 15),
            max_date_allowed=dt(2020, 6, 11),
            initial_visible_month=dt(2020, 3, 15),
            date=str(dt(2020, 2, 17, 23, 59, 59))
        )
    ], className = 'histo')


    if at == "tab-1":
        return spot_content
    elif at == "tab-2":
        return forward_content


#Callback sur la jauge du spot
@app.callback(
    Output('historical-spot', 'figure'),
    [Input('year-slider-spot', 'value')])
def update_figure(selected):
    if selected == 4:
        filtered_df = spot_history_df.iloc[45:]
    elif selected == 5:
        filtered_df = spot_history_df[73:]
    else:
        filtered_df = spot_history_df

    data = Scatter(
        x = filtered_df['Trading Day'],
        y = filtered_df['Price']
    )
    lo = Layout(
        paper_bgcolor='rgb(0, 0, 0, 0)',
        plot_bgcolor='#f2f2f2',
        transition = {'duration': 500},
        title = 'Données spot historiques'
    )
    return Figure(data = data, layout = lo)


#Callback sur le calendrier du forward
@app.callback(
    Output('historical-forward', 'figure'),
    [Input('date-picker-forward', 'date')])
def update_figure(selected_date):
    data = Scatter(
        x = forward_history_df.columns[1:5],
        y = forward_history_df.loc[forward_history_df['Trading Day'] == str(selected_date)[:10]][['Month+1', 'Month+2', 'Month+3', 'Month+4']].iloc[0],
        mode = 'markers',
        marker = dict(size = [20, 20, 20, 20])
    )
    lo = Layout(
        paper_bgcolor='rgb(0, 0, 0, 0)',
        plot_bgcolor='#f2f2f2',
        transition = {'duration': 500},
        title = 'Données forward historiques'
    )
    return Figure(data = data, layout = lo)


#Callback génération scénario
@app.callback(
    Output('scenario', 'figure'),
    [Input("random-button", "n_clicks")])
def update_scenario(n):
    k = rd.randint(4, len(strat_df) - 1)
    v_relatif = strat_df.iloc[k][1:].to_numpy()
    m = Matrices(len(v_relatif))
    volume = np.dot(m.triang_inf, v_relatif) + 0.4*np.ones_like(v_relatif)
    figure = Figure(
        data = [Scatter(
            x = scenario_df.columns,
            y = scenario_df.iloc[k-4],
            name = 'Prix',
            marker=dict(
                color='#636efa'
            )
        ),
        Scatter(
            x = strat_df.columns[1:],
            y = volume,
            name = 'Volume',
            marker = dict(
                color = 'rgb(30, 230, 30)'
            ),
            yaxis = 'y2'
        ),
        Scatter(
            x = strat_df.columns[1:],
            y = strat_df.iloc[0][1:],
            name = 'Tunnel min',
            marker = dict(
                color = 'rgb(240, 60, 60)'
            ),
            yaxis = 'y2',
            visible = 'legendonly'
        ),
        Scatter(
            x = strat_df.columns[1:],
            y = strat_df.iloc[1][1:],
            name = 'Tunnel max',
            marker = dict(
                color = 'rgb(240, 60, 60)'
            ),
            yaxis = 'y2',
            visible = 'legendonly'
        )],

        layout = Layout(
            paper_bgcolor='rgb(242, 242, 242, 0)',
            plot_bgcolor='#f2f2f2',
            transition = {'duration': 500},
            yaxis=dict(
                title='Prix €/MWh',
                titlefont=dict(
                    color='#636efa'
                ),
                tickfont=dict(
                    color='#636efa'
                )
            ),
            yaxis2 = dict(
                title='Volume normalisé',
                overlaying = 'y',
                side = 'right',
                titlefont=dict(
                    color='rgb(30, 230, 30)'
                ),
                tickfont=dict(
                    color='rgb(30, 230, 30)'
                )
            ),
            title = 'Scénario de diffusion de prix'
        )
    )
    return figure


#Callback sur le collapse de la volatilité
@app.callback(
    Output("collapse", "is_open"),
    [Input("collapse-button", "n_clicks")],
    [State("collapse", "is_open")],
)
def toggle_collapse(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=True)