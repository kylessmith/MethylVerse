# MethylVerse

[![Build Status](https://travis-ci.org/kylessmith/MethylVerse.svg?branch=master)](https://travis-ci.org/kylessmith/MethylVerse) [![PyPI version](https://badge.fury.io/py/MethylVerse.svg)](https://badge.fury.io/py/MethylVerse)
[![Coffee](https://img.shields.io/badge/-buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee&color=ff69b4)](https://www.buymeacoffee.com/kylessmith)

<img src="MethylVerse_logo.png" width="300" />
Library to work with WGBS, EM-seq, and/or methylation array data in one interface.


For full usage and installation [documentation][methylverse_docs]

## Install

If you dont already have numpy and scipy installed, it is best to download
`Anaconda`, a python distribution that has them included.  
```
    https://continuum.io/downloads
```

Dependencies can be installed by:

```
    pip install -r requirements.txt
```

PyPI install, presuming you have all its requirements installed:
```
    pip install methylverse
```

## Quick start

Test numpy random integers:

```python
import MethylVerse as mv

beta_values = mv.core.read_methylation("path/to/methylation")

```

Run the M-PACT classifier from the cammandline
```
python -m MethylVerse MPACT example.bedgraph --impute --regress --call_cnvs --verbose
```


[methylverse_docs]: https://www.biosciencestack.com/static/MethylVerse/docs/index.html