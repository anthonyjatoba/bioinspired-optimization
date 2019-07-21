# Description

This is the first project of the Bioinspired Optimization class given by Professor [Erick Barboza, PhD](https://sites.google.com/ic.ufal.br/erick/). We analyze the impact of different parameters on the Particle Swarm Optimization and Genetic Algorithm.

Students:
* [Anthony Jatobá](https://github.com/anthonyjatoba/)
* [Igor Theotônio](https://github.com/igortheotonio/)
* [Eduardo Moraes](https://github.com/dudummv/)

## Particle Swarm Optimization

The analysis can be seen in [Particle Swarm Optimization Notebook](https://github.com/anthonyjatoba/biopt/blob/master/PSO.ipynb)

## Genetic Algorithm

The analysis can be seen in [Genetic Algorithm Notebook](https://github.com/anthonyjatoba/biopt/blob/master/GA.ipynb)

## How to set the environment for the first time

Install `pip` and `virtualenv`:

`sudo apt-get install python3-pip`

`sudo pip3 install virtualenv`

Clone the project:

`git clone git@github.com:anthonyjatoba/biopt.git`

Create a virtual environment within the project directory:

`cd biopt`

`virtualenv -p python3 env`

Install the required dependencies:

`source env/bin/activate`
`pip install -r requirements.txt`

And run Jupyter Notebook:

`jupyter notebook`
