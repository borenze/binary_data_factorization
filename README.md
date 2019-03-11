## [Binary data factorization]

> This is an implementation of algorithms developped by "Mamadou". 

## Download

### Pre requierments

[Binary data factorization] works with Python 3.7+.

Dependencies:
 -   [NumPy](http://www.numpy.org)
 -   [SciPy](https://www.scipy.org)

```shell
pip install numpy scipy 
```


### Install using `pip`

You can find this package in the Python package index and install it using `pip`

```bash
pip install [project_name]
````

### Install from sources

Clone this repository

```bash
git clone https://github.com/borenze/binary_data_factorization.git
cd binary_data_factorization
```

And execute `setup.py`

```bash
pip install .
```

Of course, if you're in development mode and you want to install also dev packages, documentation and/or tests, you can do as follows:

```bash
pip install -e .
```

## Usage examples

You can import BMF by doing

```python
from codes import BMF
```

The main function included in this package is `c_pnl_pf`. `c_pnl_pf` receives a numpy matrix and the wish rank as arguments and factorize it into two binary matrices. Here is example of its usage:

```python
W, H = BMF.c_pnl_pf(X, rank=3)
```

A more detailed documentation can be found in [link].
