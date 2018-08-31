# -*- coding: utf-8 -*-

# extractors and filters.
# TODO: extractors can be what makes the function work in the way specified.
# the filters can be done.

# TODO: perhaps add a slider for jittering.


# extracts filters to show for filtering.

# TODO: define some visual elements.

# TODO: think about how these elements can be done for other settings that
# are not based on architecture search.

# this means that the files may come from somewhere else.

# TODO: check the categorical properties, right. I think it can

# I think that it should also be possible to work with different filters.

# NOTE: we can have an exploration of the models, along with some model
# serving. these applications are extremely dynamic, so they should never work.

### TODO: add some good configs to generate predefined graphs.
# TODO: add some functionality to automatically store the graph in memory.
# this is going to be interesting.
# cfg = {
#     'xaxis' : 'validation_accuracy',
#     'num_param'
# }

# buttons in dash
# wrap all these aspects to make sure that they work correctly.

# TODO: perhaps break the application into multiple cases.

# the configuration can be done easily. I think that it is OK to rely on the
# evaluation folders, even if we are looking to extend the model to other cases.

# NOTE: the repetition aspect can be done with it. save state somewhere, I guess
# in the logging folder, just some JSON file.

## TODO: add the filtering part here, to make sure that is kind of works.
# TODO: it would be convenient to define the graphical elements somewhere
# rather than reproduce them all the time.
# TODO: some of the layouts should be done for the 1D, 2D, and 3D cases.
# TODO: the choices of what models to plot comes from

# TODO: this can be simultaneously about the model and the
# have the ability of querying the model.

# arbitrary filters and arbitrary properties.
# coloring things acording to the values or to the properties.

# TODO: improve this part of the model, as I think that it can be done
# nicely. what is

# TODO: maybe change the layout with respect to the toher model.
# TODO: I can do something with the model to make sure that it works OK?

# perhaps glob the information from the network.
# todo: start a config:

# TODO: look into the drug discovery one. I think that this is going to
# be interesting.

# TODO: focus on the data that is going to work well.

# for the hover functionality it is still tricky. how to do it? what to show.
# we can simply show some information plotted, but it seems tricky.

# 3D graphs are of limited utility. prefer 2D plots, perhaps with colormaps
# and parallel coordinates (for now, only for simple examples. 3d plots are
# useful
#
# use the simple examples, also use the style of the

# 3D plots for embedding based plots. This is nice, strongly tied to

# NOTE: this is extremely dynamic. it can be used to pull information from
# the internet or just extract information from the files.

# TODO: it is cool to see how it wokr

# look into ipython for the documentation of dcc and go.

cfg = {
    'logs_folderpath' : 'logs/cifar10_medium/run-0/',
}
# NOTE: there is a more complete representation above.

# line plots for now are just going to be about the search.

# NOTE: another possibility is making sure that that the model

# {table, scatter_plot, parallel_plot}
ex = {
    'plots' : {
        'table'
    }

}

# if the weight one is not set, it does not have it.

import dash
import dash_core_components as dcc
import dash_html_components as html
import deep_architect.search_logging as sl
import deep_architect.visualization as vi

app = dash.Dash()

# read multiple folders, something like that.

log_lst = sl.read_search_folder('logs/cifar10_medium/run-0/')
ds = []
for log in log_lst:
    d = log['results']
    d.pop('sequences')
    ds.append(d)
keys = ds[0].keys()
print keys
# check that each k is a collection that can be sorted on.
### TODO: this is going to be important

# TODO: do the embedding visualization based on the model.
# It is nice that JSON is a language to describe computation.
# if there is no reason to do it in something other than JSON, just do
# it in JSON.html

# for scatter, it makes sense to have a weight key.
# everything is based on keys.file

# for scatter, it also makes sense to have multiple ones, perhaps overlapping,
# TODO: leave setting up the properties for later.

# TODO: figure out the right identation for the model.

# TODO: it is going to be important to think about the model in the sense
# there may exist the need of doing repetitions for the model in question.
# in this case, it can be useful to do error bars. ignore for now.

# have functions both for plots and a placement in the application.

# for the log ones, it may be necessary to do log(x + 1) to guarantee that it
# is a positive integer. log _10 (is probably the best one). what is the
# problem with this? makes sense for log()

# NOTE: this is the simplified representation of a page of plots.
### NOTE: this contains most of the information.
[{
    "search_name_lst" : ['cifar10_medium', 'cifar10_small'],
    "type" : 'scatter2d',
    "x" : {
        'key': 'test_accuracy',
        'type': 'log'
    },
    "y" : {
        'key': 'validation_accuracy',
        'type': 'linear'
    },
    "size" : {
        'key': 'num_parameters',
        'type': 'log'
    },
    "global_filters" : [
        {'name' : 'remove_less_than', # other analog models can also be used.
         'key' : 'validation_accuracy'}
    ],
    "name" : "first",
    "depends_on" : ['initial', 'other'],
}]

# this can be convenient to show that they depend on each other. what does this
# buy you? allows you to represent plots easily. derived plot, which can be done by
# adding some additional information.

# define a simple grammar to do plots.

# TODO: just give the number of keys necessary to construct the graph, but
# make it have an uniform interface.

# table it is going to be a similar row, or something like that.
# NOTE: check what is already implmeen

# a page is [ plots ]; where each plot is composed of multiple ones

# NOTE: filters can be applied both at the local level, and at the glo
# NOTE: the visualizations can be extremely dynamic, i.e., as long as the
# file as the keys that we need, it should just work.

# NOTE: filters can be applied globally. there can also
# be some form of simple way of writing down a graph.
# the part of the graph that is mostly dynamic is extracted to a JSON file.

# filters:
# - remove_less_or_equal_than
# - keep_less_or_equal_than
# - remove_top_k
# - keep_top_k
# - remove_bottom_k
# - keep_bottom_k

# - remove_if_in_range
# - keep_if_in_range

# - keep_random_k (seed)

# - keep_if_equal_to
# - remove_if_equal_to
# - keep

# custom filters with some arguments. can be registered and something can be
# done with them. they can be listed in the same way.
# list of available filters this is nice. Just have them somewhere.

# the question is how to show these argumernts

# dropdown to pick filter -> changes the available arguments for the filter
# along with an apply button.

# NOTE: it may be convenient to represent graphs and to represent some other models.

# remove, keep (bottom)
# ---> remove

# some simple filters.
# given all the information defined in the model, it would be interesting to
# consider the case where the models are going to work well together.

# most filters,

# easy to do plotting with a JSON language.
# legend can be the path to the folder.

###

# TODO: can models be combined? if so, what is the right behavior?
# TODO: have the ability to have text there.

# hover functionality.

#### <--->

# TODO: graph type.
### TODO: have an easy way of having columns.html
# TODO: add tables to the model.

import dash
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import pandas as pd

# TODO: perhaps have an option to add an option to switch left and right.
# this is useful to switch between different formats.
# TODO: have a way of working with text easily, e.g., what should
# be the text in there.
# this should be constrained

# multiple selectors across plots. would this be possible or not.

def full_column(contents):
    return html.Div(contents, className="row")

def one_half_one_half_column(left_contents, right_contents):
    return html.Div([
        html.Div(left_contents, className="six columns"),
        html.Div(right_contents, className="six columns"),
    ], className="row")

def one_third_two_thirds_column(left_contents, right_contents):
    return html.Div([
        html.Div(left_contents, className="four columns"),
        html.Div(right_contents, className="eight columns"),
    ], className="row")



app = dash.Dash()

app_other = dash.Dash()
app_other.layout = html.Div([
    html.Div([
        html.Div([
            html.H3('Column 1'),
            dcc.Graph(id='g1', figure={'data': [{'y': [1, 2, 3]}]})
        ], className="four columns"),

        html.Div([
            html.H3('Column 2'),
            dcc.Graph(id='g2', figure={'data': [{'y': [1, 2, 3]}]})
        ], className="eight columns"),
    ], className='row')
])

# external_css = ["https://cdnjs.cloudflare.com/ajax/libs/skeleton/2.0.4/skeleton.min.css",
#                 "//fonts.googleapis.com/css?family=Raleway:400,300,600",
#                 "//fonts.googleapis.com/css?family=Dosis:Medium",
#                 "https://cdn.rawgit.com/plotly/dash-app-stylesheets/0e463810ed36927caf20372b6411690692f94819/dash-drug-discovery-demo-stylesheet.css"]


# for css in external_css:
#     app.css.append_css({"external_url": css})

# TODO: I should probably add more ids to these elements.

# radio buttons:
def linear_vs_log_radio_items():
    return dcc.RadioItems(
            options=[
                {'label': 'linear', 'value': 'linear'},
                {'label': 'log', 'value': 'log'},
            ],
            value='linear',
            labelStyle={'display': 'inline-block'})

def dropdown(option_lst):
    return dcc.Dropdown(
            options=[{'label' : k, 'value' : k} for k in option_lst])


#### layout = go.Layout(
#     plot_bgcolor = '#E5E5E5',
#     paper_bgcolor = '#E5E5E5'
# )
# this

# it is kind of weight to represent weight if it has not been used to
# compute the embeddings. I think that it is best to just keep it simple for
# now and then as we go, to add more functionality.

# TODO: set the background color and the paper color appropriately.

# TODO: there are asserts to deal with other type of data.
# TODO: change the properties of these models.
# TODO: how to do multiple simulations at once.

# TODO: simplify this. I think that it is fine to change it latter.
def scatter1d(ds, key_x):
    return dcc.Graph(
        # id='arch-1d',
        figure={
            'data': [
                go.Scatter(
                    x=[d[key_x] for d in ds],
                    y=[0.0] * len(ds),
                    # text=df[df['continent'] == i]['country'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                )
            ],
            'layout': go.Layout(
                xaxis={
                    # 'type': 'log',
                    'title': key},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )




# TODO: it is going to insert these models. I think that this is interesting.

# TODO: can manage multiple ones.
def scatter2d(graph_id, ds, key_x, key_y):
    return dcc.Graph(
        id='life-exp-vs-gdp',
        figure={
            'data': [
                go.Scatter(
                    x=df[df['continent'] == i]['gdp per capita'],
                    y=df[df['continent'] == i]['life expectancy'],
                    text=df[df['continent'] == i]['country'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                    name=i
                ) for i in df.continent.unique()
            ],
            'layout': go.Layout(
                xaxis={'type': 'log', 'title': 'GDP Per Capita'},
                yaxis={'title': 'Life Expectancy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )

## TODO: add some
def search_scatter2d(graph_id, ds, key_x, key_y):
    return dcc.Graph(
        id='arch-1d-over-time',
        figure={
            'data': [
                go.Scatter(
                    x=range(1, len(ds) + 1),
                    y=[d['validation_accuracy'] for d in ds],
                    # text=df[df['continent'] == i]['country'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                ),
                go.Scatter(
                    x=range(1, len(ds) + 1),
                    y=vi.running_max([d['validation_accuracy'] for d in ds]),
                    # text=df[df['continent'] == i]['country'],
                    mode='line',
                    # opacity=0.7,
                    # marker={
                    #     'size': 15,
                    #     'line': {'width': 0.5, 'color': 'white'}
                    # },
                )
            ],
            'layout': go.Layout(
                xaxis={
                    # 'type': 'log',
                    'title': 'evaluation_number'},
                yaxis={
                    # 'type': 'log',
                    'title': 'validation_accuracy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )

# TODO: add information about the ID and other things.
def parallel_plot():
    return dcc.Graph(
        id='arch-parallel',
        figure={
            'data': [
                go.Parcoords(
                    line = dict(color = 'blue'),
                    dimensions = list([
                        dict(
                            range = [1,5],
                            # constraintrange = [1,2],
                            label = 'A',
                            values = [1,4]
                            ),
                        dict(range = [1.5,5],
                            # tickvals = [1.5,3,4.5],
                            label = 'B',
                            # values = [3,1.5]
                            ),
                        dict(range = [1,5],
                            # tickvals = [1,2,4,5],
                            label = 'C', values = [2,4],
                            # ticktext = ['text 1', 'text 2', 'text 3', 'text 4']
                            ),
                        dict(range = [1,5],
                            label = 'D',
                            values = [4,2])
                    ])
                )
            ], ### NOTE: layout is just a way of adding information about the
            'layout': go.Layout(
                xaxis={
                    # 'type': 'log',
                    'title': 'validation_accuracy'},
                yaxis={
                    # 'type': 'log',
                    'title': 'num_parameters'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    )

# TODO: add extractors that are then available.

# I guess that some of these functions can be changed.

# I need to undestand this better.

app_other.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})


df = pd.read_csv(
    'https://gist.githubusercontent.com/chriddyp/' +
    '5d1ea79569ed194d432e56108a04d188/raw/' +
    'a9f9e8076b837d541398e999dcbac2b2826a81f8/'+
    'gdp-life-exp-2007.csv')

# TODO: add some elements to make it easy to draw these elements.

# TODO: it should be easy to change the result upon the addition of a new one.


app.layout = html.Div([
    html.Div([
        dcc.RadioItems(
            options=[
                {'label': 'linear', 'value': 'linear'},
                {'label': 'log', 'value': 'log'},
            ],
            value='linear',
            labelStyle={'display': 'inline-block'}
        ),
        dcc.Dropdown(
            options=[{'label' : k, 'value' : k} for k in ds[0]]
        ),
        dcc.Dropdown(
            options=[{'label' : k, 'value' : k} for k in ds[0]]
        ),
        dcc.Dropdown(
            options=[{'label' : k, 'value' : k} for k in ds[0]]
        )
    ]),
    one_third_two_thirds_column([
        html.H1('Hi')
    ], [
        dcc.Graph(
            id='arch-performance',
            figure={
                'data': [
                    go.Scatter(
                        x=[d['validation_accuracy'] for d in ds],
                        y=[d['num_parameters'] for d in ds],
                        # text=df[df['continent'] == i]['country'],
                        mode='markers',
                        opacity=0.7,
                        marker={
                            'size': 15,
                            'line': {'width': 0.5, 'color': 'white'}
                        },
                    )
                ],
                'layout': go.Layout(
                    xaxis={
                        # 'type': 'log',
                        'title': 'validation_accuracy'},
                    yaxis={
                        'type': 'log',
                        'title': 'num_parameters'},
                    margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                    legend={'x': 0, 'y': 1},
                    hovermode='closest'
                )
            }
        ),
    ]),
    dcc.Graph(
        id='arch-3d',
        figure={
            'data': [
                go.Scatter3d(
                    x=[d['validation_accuracy'] for d in ds],
                    y=[d['num_parameters'] for d in ds],
                    z=[d['inference_time_per_example_in_miliseconds'] for d in ds],
                    # text=df[df['continent'] == i]['country'],
                    mode='markers',
                    # opacity=0.7,
                    # marker={
                    #     'size': 15,
                    #     'line': {'width': 0.5, 'color': 'white'}
                    # },
                )
            ],
            'layout': go.Layout(
                scene=go.Scene(
                    xaxis={
                        # 'type': 'log',
                        'title': 'validation_accuracy'},
                    yaxis={
                        'type': 'log',
                        'title': 'inference_time_per_example_in_miliseconds'},
                    zaxis={
                        # 'type': 'log',
                        'title': 'num_parameters'}
                ),
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                # legend={'x': 0, 'y': 1, 'z': 2},
                # hovermode='closest'
            )
        }
    ),
    dcc.Graph(
        id='arch-1d',
        figure={
            'data': [
                go.Scatter(
                    x=[d['validation_accuracy'] for d in ds],
                    y=[0.0] * len(ds),
                    # text=df[df['continent'] == i]['country'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                )
            ],
            'layout': go.Layout(
                xaxis={
                    # 'type': 'log',
                    'title': 'validation_accuracy'},
                yaxis={
                    # 'type': 'log',
                    'title': 'num_parameters'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    ),
    dcc.Graph(
        id='arch-1d-over-time',
        figure={
            'data': [
                go.Scatter(
                    x=range(1, len(ds) + 1),
                    y=[d['validation_accuracy'] for d in ds],
                    # text=df[df['continent'] == i]['country'],
                    mode='markers',
                    opacity=0.7,
                    marker={
                        'size': 15,
                        'line': {'width': 0.5, 'color': 'white'}
                    },
                ),
                go.Scatter(
                    x=range(1, len(ds) + 1),
                    y=vi.running_max([d['validation_accuracy'] for d in ds]),
                    # text=df[df['continent'] == i]['country'],
                    mode='line',
                    # opacity=0.7,
                    # marker={
                    #     'size': 15,
                    #     'line': {'width': 0.5, 'color': 'white'}
                    # },
                )
            ],
            'layout': go.Layout(
                xaxis={
                    # 'type': 'log',
                    'title': 'evaluation_number'},
                yaxis={
                    # 'type': 'log',
                    'title': 'validation_accuracy'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    ),
    dcc.Graph(
        id='arch-parallel',
        figure={
            'data': [
                go.Parcoords(
                    line = dict(color = 'blue'),
                    dimensions = list([
                        dict(
                            range = [1,5],
                            # constraintrange = [1,2],
                            label = 'A',
                            values = [1,4]
                            ),
                        dict(range = [1.5,5],
                            # tickvals = [1.5,3,4.5],
                            label = 'B',
                            # values = [3,1.5]
                            ),
                        dict(range = [1,5],
                            # tickvals = [1,2,4,5],
                            label = 'C', values = [2,4],
                            # ticktext = ['text 1', 'text 2', 'text 3', 'text 4']
                            ),
                        dict(range = [1,5],
                            label = 'D',
                            values = [4,2])
                    ])
                )
            ],
            'layout': go.Layout(
                xaxis={
                    # 'type': 'log',
                    'title': 'validation_accuracy'},
                yaxis={
                    # 'type': 'log',
                    'title': 'num_parameters'},
                margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
                legend={'x': 0, 'y': 1},
                hovermode='closest'
            )
        }
    ),
    # html.Div(children='''
    #     Dash: A web application framework for Python.
    # ''')
])

# how to add layout to the parallel coordinates.

# trace1 = Scatter3d(
#     x=[1, 2],
#     y=[1, 2],
#     z=[1, 2]
# )
# data = Data([trace1])
# layout = Layout(
#     scene=Scene(
#         xaxis=XAxis(title='x axis title'),
#         yaxis=YAxis(title='y axis title'),
#         zaxis=ZAxis(title='z axis title')
#     )
# )
# fig = Figure(data=data, layout=layout)

if __name__ == '__main__':
    app.run_server(debug=True)
    # app_other.run_server(debug=True)

