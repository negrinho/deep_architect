

Callback description:

Describes what each callback does, what information it consumes and how does it
interact with the other callbacks.

TODO: split things into categories.

## Graph update (scatter2d_update_callback):

Inputs:
* Value of the log selector dropdown (multi-valued).
* Key to plot on the x axis.
* Key to plot on the y axis.
* Scale of the x axis.
* Scale of the y axis.

Output:
* Dictionary with data and layout information.

TODO: will depend on the configuration of the filters.

## Add row (_add_row_callback):

Inputs:
* Number of clicks to the add row button.

Output:
* Contents of the div with the rows so far

State:
* Contents of the div with the rows so far

TODO: it needs to be changed in a way that does not keep state, but it still
goes through the current contents of the page. Can exist at the bottom of the page.

## Delete row (_delete_row_callback)

Inputs:
* Number of clicks to the add row button.

Output:
* Contents of the div with the rows so far

State:
* Contents of the div with the rows so far

TODO: row specific positioning.

## Copy row (_copy_row)

Input:

Output:

State:

## Save state (save_state?):

Inputs:
* Number of clicks of the save button

State:
* Value of the text box on where to save to.

NOTE: this can have an additional output, by having the model update the values of
the available files to load from. Maybe they can be saved to some reasonable folder.
These can be directly available during the beginning.
Maybe these can be done independently.

## Load state (load_state):

Inputs:
* Number of clicks to the load state button

State:
* Value of component that needs to be saved.


TOOD: it really just needs the contents of the visualization and should go from there.
the indexing can be done.
Perhaps this can specified upon creation of the server (i.e., what folder should these
files be saved to).
Put the results of the loading a file in an hidden div that can be used as caching.

## Apply filters (apply_filters)

Inputs:
* Values of the dropdowns (many per row)

State:
* Current value of the div with the dropdowns that are visible.


Create a function to cache information in the page.

Tips:

* To debug, don't hide the hidden div. How to embed data into it? How to embed JSON
into HTML components. Look this up.
* How to do caching in the case of expensive operations?
* No Python state mutation after the server starts running. All Python functions
serve solely as auxiliary functions to make it easier to address the relevant fields
in the webpage.
* State is used if I want to communicate some information to the callback. Inputs
are used to make the callback trigger when the value of that field changes. Outputs
are used to communicate changes to that field, which are computed based on the
inputs and the state.
* Certain state is global like the data, which can be cached directly, or loaded
lazily. Other things are limits to the maximum number of certain elements in
the visualization.
* Some of these are row specific.
* Don't forget the divs or just access them directly.


