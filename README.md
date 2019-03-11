## [Binary data factorization]

> This is an implementation of algorithms developped by "Mamadou". 

## Download

### Pre requierments

[Binary data factorization] works with Python 3.7+.

Dependencies:
 -   [NumPy](http://www.numpy.org)
 -   [SciPy](https://www.scipy.org)
 -   [Scikit-learn](https://scikit-learn.org/stable/index.html)
 

```shell
pip install numpy scipy scikit-learn
```


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


