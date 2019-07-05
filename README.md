## Como configurar o ambiente pela primeira vez

Instale o `pip` e o `virtualenv`:

`sudo apt-get install python3-pip`
`sudo pip3 install virtualenv`

Clone o projeto:

`git clone git@github.com:anthonyjatoba/biopt.git`

Crie um ambiente virtual dentro do repositório com o virtualenv:

`cd biopt`
`virtualenv -p python3 env`

Instale as dependências necessárias para a execução dos experimentos:

`source env/bin/activate`
`pip install -r requirements.txt`

E execute o Jupyter Notebook:

`jupyter notebook`
