# MethylVerse

[![Build Status](https://travis-ci.org/kylessmith/MethylVerse.svg?branch=master)](https://travis-ci.org/kylessmith/MethylVerse) [![PyPI version](https://badge.fury.io/py/MethylVerse.svg)](https://badge.fury.io/py/MethylVerse)
[![Coffee](https://img.shields.io/badge/-buy_me_a%C2%A0coffee-gray?logo=buy-me-a-coffee&color=ff69b4)](https://www.buymeacoffee.com/kylessmith)

<img src="MethylVerse_logo.png" width="300" />
Library to work with WGBS, EM-seq, and/or methylation array data in one interface.


## Install

If you dont already have numpy and scipy installed, it is best to download
`Anaconda`, a python distribution that has them included.  
```
    https://continuum.io/downloads
```

PyPI install, presuming you have all its requirements installed:
```
    pip install ngsfragments
	pip install MethylVerse
```

The developmental version of MethylVerse can also be install from GitHub.
```
    pip install git+https://github.com/kylessmith/MethylVerse/
```

First use of the MethylVerse software will initialize a download of necessary reference
data. Raw reference data is also available at on Zenodo.
```
    https://zenodo.org/records/16580408
    https://zenodo.org/records/16581863
```


## Usage

General useage:

```python
import MethylVerse as mv

beta_values = mv.core.read_methylation("path/to/methylation")

classifier = mv.tools.classifiers.MPACT.MPACT_classifier_torch.MPACT_classifier()
predictions = classifier.predict(beta_values)

```

Run the M-PACT classifier from the cammandline
```
python -m MethylVerse MPACT example.bedgraph --impute --regress --call_cnvs --verbose
```

Additional options can be viewed:
```
python -m MethylVerse MPACT -h

usage: __main__.py MPACT [-h] [--impute] [--regress] [--probability_threshold PROBABILITY_THRESHOLD]
                         [--max_contamination_fraction MAX_CONTAMINATION_FRACTION] [--call_cnvs] [--out OUT] [--verbose]
                         input_data

positional arguments:
  input_data            Input data, idat name, nanopore bed file, or MethylDackel bedGraph

options:
  -h, --help            show this help message and exit
  --impute              Impute data missing CpGs
  --regress             Whether to regress data and remove background CSF/Immune
  --probability_threshold PROBABILITY_THRESHOLD
                        Probability threshold for M-PACT classification (default: 0.7)
  --max_contamination_fraction MAX_CONTAMINATION_FRACTION
                        Max contamination fraction for M-PACT classification (aggressiveness of removing background, default=0.3)
  --call_cnvs           Call CNVs from the methylation file
  --out OUT             Output file
  --verbose             Verbose output
```

[methylverse_docs]: https://www.biosciencestack.com/static/MethylVerse/docs/index.html