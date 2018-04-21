
# -*- coding: utf-8 -*-

import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output, State
import plotly.graph_objs as go

import darch.search_logging as sl

### Loading the data.
def process_logs(log_lst):
    ds = []
    for i, log in enumerate(log_lst):
        d = log['results']
        # d.pop('sequences')
        # d['num_training_epochs'] = len(d['num_training_epochs'])
        d['evaluation_id'] = i
        ds.append(d)
    return ds

# path_lst = ['logs/test_cifar10_short', 'logs/test_cifar10_medium', 'logs/test']
path_lst = ['logs/test_cifar10_medium', 'logs/test_cifar10_short']
path_to_log = {p : process_logs(sl.read_search_folder(p)) for p in path_lst}
keys = path_to_log.values()[0][0].keys()
print keys
ds = path_to_log.values()[0]

# NOTE: it depends if I pass the path directly or something different.
# NOTE: some idea of post-processing can be done here.
# these are all options. it should correspond to actual
class LogManager:
    def __init__(self, log_folderpath_lst):
        pass

    # NOTE: this is
    def get_data(self, log_folderpath, key_lst):
        pass

    def get_shared_keys(self, log_folderpath_lst):
        pass

# the actual keys that may be available may be managed by the log manager.

# this class helps interfacing with the logs, which makes it a lot easier to
# retrieve the desired data, and apply the filters.
# the all data, is passed and the filter needs to return a mask over the elements.
# composition.
# do some form of filter composition.

def maybe_swap(a, b, swap):
    return (b, a) if swap else (a, b)

# TODO: add swapping functionality to easily try different layouts.
#### the stuff above is on progress.

### Components to layout the HTML page.
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

def two_thirds_one_third_column(left_contents, right_contents):
    return html.Div([
        html.Div(left_contents, className="eight columns"),
        html.Div(right_contents, className="four columns"),
    ], className="row")

def horizontal_separator():
    return html.Hr()

### Abstract class for a nicer naming experience.
class Component:
    def __init__(self, parent_name, local_name):
        self.local_name = local_name
        self.parent_name = parent_name
        self.full_name = parent_name + '/' + local_name
        self.html_comp = None

    def get_local_name(self):
        return self.local_name

    def get_parent_name(self):
        return self.parent_name

    def get_full_name(self):
        return self.full_name

    def get_input(self, attribute_name):
        return Input(self.full_name, attribute_name)

    def get_output(self, attribute_name):
        return Output(self.full_name, attribute_name)

    def get_state(self, attribute_name):
        return State(self.full_name, attribute_name)

    def get_layout(self):
        return self.html_comp

    def _register(self, html_comp):
        assert self.html_comp is None
        self.html_comp = html_comp

### General components that are used throughout many dashboards of the visualization.
class FullColumn(Component):
    def __init__(self, parent_name, local_name, contents):
        Component.__init__(self, parent_name, local_name)
        self._register(
            html.Div(
                id=self.full_name,
                children=contents,
                className="row"))
        # children is useful as an output for callbacks.

class Dropdown(Component):
    def __init__(self, parent_name, local_name, option_lst, placeholder_text, is_multi):
        Component.__init__(self, parent_name, local_name)
        self._register(
            dcc.Dropdown(
                id=self.full_name,
                options=[{'label' : k, 'value' : k} for k in option_lst],
                placeholder=placeholder_text,
                multi=is_multi))
        # value is useful to create callbacks for other aspects.

    def get_value_input(self):
        return self.get_input('value')

class Button(Component):
    def __init__(self, parent_name, local_name, buttom_text):
        Component.__init__(self, parent_name, local_name)
        self._register(
            html.Button(
                id=self.full_name,
                children=buttom_text))
        # n_clicks if the property that should be used for on click events.

    def get_nclicks_input(self):
        return self.get_input('n_clicks')

class RadioItems(Component):
    def __init__(self, parent_name, local_name, option_lst, is_inline):
        Component.__init__(self, parent_name, local_name)
        self._register(
            dcc.RadioItems(
                id=self.full_name,
                options=[{'label': v, 'value': v} for v in option_lst],
                # value=option_lst[0],
                labelStyle={'display': 'inline-block'} if is_inline else None))

    def get_value_input(self):
        return self.get_input('value')

class Text(Component):
    def __init__(self, parent_name, local_name, str_value):
        Component.__init__(self, parent_name, local_name)
        self._register(
            html.P(
                id=self.full_name,
                children=str_value))
        # children is perhaps useful if we want to change the text.

class TextInput(Component):
    def __init__(self, parent_name, local_name, placeholder_text):
        Component.__init__(self, parent_name, local_name)
        self._register(
            dcc.Input(
                id=self.full_name,
                placeholder=placeholder_text,
                type='text',
                value='',
                style={'width': '100%'}))
        # value is an input attribute to read from callbacks.

    def get_value_input(self):
        return self.get_input('value')

class TextBox(Component):
    def __init__(self, parent_name, local_name, placeholder_text):
        Component.__init__(self, parent_name, local_name)
        self._register(
            dcc.Textarea(
                id=self.full_name,
                placeholder=placeholder_text,
                style={'width': '100%'}))

class Notes(Component):
    def __init__(self, parent_name, local_name):
        Component.__init__(self, parent_name, local_name)
        self._register(
            full_column([
                Text(self.full_name, 'title', 'Notes').get_layout(),
                TextBox(self.full_name, 'description',
                    'Write your notes about this row here.').get_layout()
            ]))

class LogSelectorDropdown(Component):
    def __init__(self, parent_name, local_name):
        Component.__init__(self, parent_name, local_name)
        self.dropdown = Dropdown(parent_name, local_name, path_to_log.keys(),
            'Choose search logs to visualize.', True)
        self._register(
            full_column([
                Text(self.full_name, 'title', 'Search logs').get_layout(),
                self.dropdown.get_layout()
            ]))

    def get_value_input(self):
        return self.dropdown.get_value_input()

# NOTE: the log scale has not been very useful now.
class DimensionControls(Component):
    def __init__(self, parent_name, local_name, dimension_selector_placeholder_text):
        Component.__init__(self, parent_name, local_name)
        self.dropdown = Dropdown(self.full_name, 'selector', ds[0].keys(),
            dimension_selector_placeholder_text, False)
        self.radio = RadioItems(self.full_name, 'scale', ['linear', 'log'], True)
        self._register(
            two_thirds_one_third_column([
                self.dropdown.get_layout()
            ],[
                self.radio.get_layout()
            ]))

    def get_value_input_lst(self):
        return [self.dropdown.get_value_input(), self.radio.get_value_input()]

# NOTE: can be transformed to something that is similar to other dimensions.
class Scatter2DDimensionControls(Component):
    def __init__(self, parent_name, local_name):
        Component.__init__(self, parent_name, local_name)
        self.x_controls = DimensionControls(self.full_name, 'x', 'Horizontal axis')
        self.y_controls = DimensionControls(self.full_name, 'y', 'Vertical axis')
        # self.marker_size_controls = DimensionControls(self.full_name, 'marker_size', 'Marker size axis')
        self._register(
            full_column([
                Text(self.full_name, 'title', 'Dimensions').get_layout(),
                self.x_controls.get_layout(),
                self.y_controls.get_layout(),
                # self.marker_size_controls.get_layout(),
            ]))

    def get_value_input_lst(self):
        return self.x_controls.get_value_input_lst() + self.y_controls.get_value_input_lst()

# TODO: change this name to dashboard.
class Scatter2DControls(Component):
    def __init__(self, parent_name, local_name):
        Component.__init__(self, parent_name, local_name)
        self.log_selector = LogSelectorDropdown(self.full_name, 'log_selector')
        self.dimension_controls = Scatter2DDimensionControls(self.full_name, 'dimension_controls')
        # self.filter_selector = FilterSelectorDropdown(self.full_name, 'filter_selector')
        self.notes = Notes(self.full_name, 'notes')
        self._register(
            full_column([
                self.notes.get_layout(),
                # horizontal_separator(),
                self.log_selector.get_layout(),
                # horizontal_separator(),
                self.dimension_controls.get_layout(),
                # horizontal_separator(),
                # self.filter_selector.get_layout(),
            ]))

    def get_update_figure_input_lst(self):
        return ([self.log_selector.get_value_input()] +
            self.dimension_controls.get_value_input_lst())

class Scatter2D(Component):
    def __init__(self, parent_name, local_name):
        Component.__init__(self, parent_name, local_name)

        self._register(
            dcc.Graph(
                id=self.full_name,
                figure={
                    'data': [],
                    'layout': make_layout(None, None, None, None)
                }))

    def get_figure_output(self):
        return self.get_output('figure')

class Scatter2DRow(Component):
    def __init__(self, parent_name, local_name):
        Component.__init__(self, parent_name, local_name)

        self.graph_controls = Scatter2DControls(self.full_name, 'graph_controls')
        self.graph = Scatter2D(self.full_name, 'graph')
        self._register(
            one_third_two_thirds_column([
                    self.graph_controls.get_layout()
                ],[
                    self.graph.get_layout()]))

    def get_callback_config_lst(self):
        output = self.graph.get_figure_output()
        input_lst = self.graph_controls.get_update_figure_input_lst()
        state_lst = []
        callback_fn = scatter2d_update_callback
        return [(output, input_lst, state_lst, callback_fn)]

def make_scatter2d(x, y):
    opts_dict = {
        'mode' : 'markers',
        # 'opacity' : 0.7,
        # 'marker' : {
        #     'size': 15,
        #     'line': {'width': 0.5, 'color': 'white'}
        # }
    }
    return go.Scatter(
            x=x,
            y=y,
            text=['x%d' % i for i in xrange(len(x))],
            **opts_dict)

def make_layout(xkey, xscale, ykey, yscale):
    opts_dict = {
        'margin' : {'l': 40, 'b': 40, 't': 10, 'r': 10},
        'legend' : {'x': 0, 'y': 1},
        'hovermode' : 'closest'
    }
    return go.Layout(
        xaxis={'type': xscale, 'title': xkey},
        yaxis={'type': yscale, 'title': ykey},
        **opts_dict)

def scatter2d_update_callback(log_folderpath_lst, xkey, xscale, ykey, yscale):
    data = []
    for path in log_folderpath_lst:
        x = []
        y = []
        for d in path_to_log[path]:
            x.append(d[xkey])
            y.append(d[ykey])
        data.append(make_scatter2d(x, y))
    return {
        'data' : data,
        'layout' : make_layout(xkey, xscale, ykey, yscale)
    }

# class GraphRow(Component):
#     def __init__(self, parent_name, local_name, graph_type):
#         Component.__init__(self, parent_name, local_name)
#         if graph_type == 'scatter2d':
#             self.graph_controls = Scatter2DControls(self.full_name, 'graph_controls')
#             self.graph = Scatter2D(self.full_name, 'graph')
#         else:
#             raise ValueError
#         self._register(
#             one_third_two_thirds_column([
#                     self.graph_controls.get_layout()
#                 ],[
#                     self.graph.get_layout()]))

# class RowControlButtons(Component):
#     def __init__(self, parent_name, local_name, graph_type):
#         Component.__init__(self, parent_name, local_name)
#         self.delete_button = Button(self.full_name, 'delete_button', 'Delete')
#         self.delete_button = Button(self.full_name, 'delete_button', 'Delete')

## TODO: add the remove button.
# class RowButtonControls(Component):
#     def __init__(self, parent_name, local_name):
#         Component.__init__(self, parent_name, local_name)

### High-level dashboards made out of the more general components.
class Header(Component):
    def __init__(self, parent_name, local_name):
        Component.__init__(self, parent_name, local_name)
        self.title = Text(self.full_name, 'title', 'Summary')
        self.description = TextBox(self.full_name, 'description',
            'Write your summary about the visualization here.')
        self._register(
            full_column([
                self.title.get_layout(),
                self.description.get_layout()
            ]))

class Footer(Component):
    def __init__(self, parent_name, local_name):
        Component.__init__(self, parent_name, local_name)
        # self.add_row_dropdown = Dropdown(self.full_name, 'add_row_dropdown',
        #     'Select row type.')
        self.add_row_button = Button(self.full_name, 'add_button', 'Add Row')
        # self.copy_row_dropdown = Dropdown(self.full_name, )
        self._register(
            full_column([
                # Text(self.full_name, 'title', 'Row').get_layout(),
                self.add_row_button.get_layout()
            ]))

    def get_add_button_nclicks_input(self):
        return self.add_row_button.get_nclicks_input()

# NOTE: this implementation is kind of weird because it constructs all possible
# rows upfront.
class Visualization(Component):
    def __init__(self, parent_name, local_name, max_num_rows):
        Component.__init__(self, parent_name, local_name)
        self.header = Header(self.full_name, 'header')
        self.footer = Footer(self.full_name, 'footer')

        self.id_to_row = {i : Scatter2DRow(self.full_name, 'row%d' % i)
            for i in xrange(max_num_rows)}
        self.used_row_id_lst = []
        self.row_column = FullColumn(self.full_name, 'row_list', [])

        # create the page layout.
        lst = [
            self.header.get_layout(),
            # horizontal_separator(),
            self.row_column.get_layout(),
            self.footer.get_layout()
            ]
        # NOTE: can be adapted later.
        # for i in self.used_row_id_lst:
        #     row = self.id_to_row[i]
        #     lst.append(row.get_layout())
        #     lst.append(horizontal_separator())
        # lst.append()

        self._register(full_column(lst))

    def _add_row_callback(self, n_clicks, rows):
        print self.used_row_id_lst, n_clicks
        if n_clicks > 0:
            for i in self.id_to_row:
                if i not in self.used_row_id_lst:
                    self.used_row_id_lst.append(i)
                    new_row = self.id_to_row[i]
                    rows.append(new_row.get_layout())
                    # rows.append(horizontal_separator())
                    break
        else:
            self.used_row_id_lst = []
        return rows

    def _get_add_row_callback_config_lst(self):
        output = self.row_column.get_output('children')
        input_lst = [self.footer.get_add_button_nclicks_input()]
        state_lst = [self.row_column.get_state('children')]
        callback_fn = self._add_row_callback
        return [(output, input_lst, state_lst, callback_fn)]

    def get_callback_config_lst(self):
        lst = []
        lst.extend(self._get_add_row_callback_config_lst())
        for row in self.id_to_row.itervalues():
            lst.extend(row.get_callback_config_lst())
        return lst

# NOTE: undo redo is also not working properly and needs to be fixed.

## NOTE: what should the starting layout be? I think that this is probably not
# quite correct.

# TODO: this does not handle correctly reloading the page. this is because of
# the none keys and what not. handle these more carefully.
# TODO: improve aesthetics. free space and other things.

# class RowList(Component):
#     def __init__(self, parent_name, local_name):
#         Component.__init__(self, parent_name, local_name)
#         self.row_lst = []
#         self.column = FullColumn(self.full_name, )


#         self._register([
#             for row in row_lst
#         ])

# should these functions be inherently tied to the object.

# get the simple callbacks done.

# ### TODO: graph controls can easily become graph something else. to include
# # comments and summary.
# ### TODO: I think that this needs to be improved
# # NOTE: just local for now.
# # the application of it changes the items

# class AppliedFilterItem(Component):
#     def __init__(self, parent_name, local_name, ):
#         Component.__init__(self, parent_name, local_name)

# class AppliedFilterList(Component):
#     def __init__(self, parent_name, local_name):
#         Component.__init__(self, parent_name, local_name)
#         self.applied_filters = []

#         self._register(
#             full_column([
#                 Text(self.full_name, 'title', 'Applied filters').get_layout(),
#                 self.filter_dropdown.get_layout()
#             ]))

# # filter selector + filter arguments.
# class FilterSelectorDropdown(Component):
#     def __init__(self, parent_name, local_name):
#         Component.__init__(self, parent_name, local_name)
#         self.filter_dropdown = Dropdown(self.full_name, 'filter_dropdown',
#             ds[0].keys(), 'Select a filter...', True)

#         self.filter_dropdown = Dropdown(self.full_name, 'filter_dropdown',
#             ds[0].keys(), 'Select a filter...', True)
#         self.filter_args = TextInput(self.full_name, 'filter_args', 'Filter arguments')
#         self.filter_apply = Button(self.full_name, 'apply_button', 'Apply')
#         self.applied_filters =


#         self._register(
#             full_column([
#                 Text(self.full_name, 'title', 'Filtering').get_layout(),
#                 self.filter_dropdown.get_layout()
#             ]))

# # TODO: can have a dropdown.
# # exposing inputs and outputs. this is going to be important.
# class FilterRegister(Component):
#     def __init__(self, parent_name, local_name):
#         Component.__init__(self, parent_name, local_name)
#         self.registered_filters = []
#         ### TODO: needs to have some form of models .

#     def register(self, filter_name, filter_args):
#         pass

#     def get_layout(self):
#         return

# TODO: header needs some extra information.

app = dash.Dash()

vis = Visualization('', 'visualization', max_num_rows=10)
app.layout = vis.get_layout()
app.config['suppress_callback_exceptions'] = True

# adding the callbacks programmatically
for (output, input_lst, state_lst, fn) in vis.get_callback_config_lst():
    app.callback(output, input_lst, state_lst)(fn)

app.css.append_css({
    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
})

if __name__ == '__main__':
    app.run_server(debug=True)

# NOTE: should never get none. only when all of them are filled.

# useful for hover events.
# Keyword arguments:
#  |  - id (string; required)
#  |  - clickData (dict; optional): Data from latest click event
#  |  - hoverData (dict; optional): Data from latest hover event
#  |  - clear_on_unhover (boolean; optional): If True, `clear_on_unhover` will clear the `hoverData` property
#  |  when the user "unhovers" from a point.
#  |  If False, then the `hoverData` property will be equal to the
#  |  data from the last point thqat was hovered over.
#  |  - selectedData (dict; optional): Data from latest select event
#  |  - relayoutData (dict; optional): Data from latest relayout event which occurs
#  |  when the user zooms or pans on the plot

# NOTE: one possibility is having the choice of logs determine the keys that are
# available. this means that there may exist interaction between the model]
# and something else.