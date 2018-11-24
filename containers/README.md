Currently, we use the Docker images from [deepo](https://github.com/ufoym/deepo).
We convert these Docker images to Singularity images.
See [here](https://github.com/ufoym/deepo#tags) for the available
tags.
We use the four images resulting from the possible choices of CPU/GPU and
Python 2.7/Python 3.6.
The corresponding tags are `all-py27`, `all-py27-cpu`, `all-py36`, `all-py36-cpu`.

To build the Singularity images, it is necessary to install Singularity, which
requires an Ubuntu operating system.
* Install instructions for [Ubuntu](http://singularity.lbl.gov/install-linux).
* Install instructions for [Windows](http://singularity.lbl.gov/install-windows).
* Install instructions for [Mac](http://singularity.lbl.gov/install-mac).

The easiest way to build images in systems other than Ubuntu is through a
virtual machine.
See [here](http://singularity.lbl.gov/install-mac#option-1-singularityware-vagrant-box)
for how to get a virtual image for Singularity on Mac.
The suggested approach uses Vagrant to setup the image.
See [here](https://www.vagrantup.com/docs/installation/)
for the instructions on how to install Vagrant.

We created a Singularity and Docker recipe file generator to create the different
containers. Run `python containers/main.py` to create the folders with the recipes
for the containers. To build the desired container run its corresponding `build.sh`
script that lies in the same folder as the recipe. All commands should
be ran from root folder of the project, i.e., same as it is used for running examples.
Additionally, we also generate build scripts in the `containers` folder for
sets of containers, e.g., all containers or containers that use Python 2.7.

To run a Singularity container with GPU support, use the `--nv` flag, e.g.,
`singularity shell --nv py27-gpu.img`.
Check the documentation of [Singularity](http://singularity.lbl.gov/docs-usage)
for more information on how to run and use Singularity containers.

**Important:** The most battle-tested containers are the Python 2.7 Singularity
ones. Expect adventures for the other ones.