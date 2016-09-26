#!/usr/bin/env python
# -*- coding: utf-8 -*-
# emacs: -*- mode: python; py-indent-offset: 4; indent-tabs-mode: nil -*-
# vi: set ft=python sts=4 ts=4 sw=4 et:
"""
Base module variables
"""
from __future__ import absolute_import, division, print_function, unicode_literals
from datetime import datetime

__packagename__ = 'diffantom'
__version__ = '1.0.2a0'
__author__ = 'Oscar Esteban'
__affiliation__ = 'Psychology Department, Stanford University'
__credits__ = ['Oscar Esteban']
__license__ = 'MIT License'
__maintainer__ = 'Oscar Esteban'
__email__ = 'code@oscaresteban.es'
__status__ = 'Prototype'
__copyright__ = 'Copyright {}, {}'.format(datetime.now().year, __author__)

__description__ = """\
Diffantom: Whole-Brain Diffusion MRI Phantoms Derived from Real Datasets of the \
Human Connectome Project"""
__longdesc__ = """\
Diffantom is a whole-brain diffusion MRI (dMRI) phantom publicly available through the \
Dryad Digital Repository (doi:10.5061/dryad.4p080). The dataset contains two single-shell \
dMRI images, along with the corresponding gradient information, packed following the BIDS \
standard (Brain Imaging Data Structure, Gorgolewski et al., 2015). \
The released dataset is designed for the evaluation of the impact of susceptibility \
distortions and benchmarking existing correction methods.\

This project contains the software instruments involved in generating \
diffantoms, so that researchers are able to generate new phantoms derived \
from different subjects, and apply these data in other applications like \
investigating diffusion sampling schemes, the assessment of dMRI processing methods, \
the simulation of pathologies and imaging artifacts, etc. In summary, Diffantom is \
intended for unit testing of novel methods, cross-comparison of established methods, \
and integration testing of partial or complete processing flows to extract connectivity \
networks from dMRI.
"""

CLASSIFIERS = [
    'Development Status :: 4 - Beta',
    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Image Recognition',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 2.7',
]

DOWNLOAD_URL = (
    'https://pypi.python.org/packages/source/{name[0]}/{name}/{name}-{ver}.tar.gz'.format(
        name=__packagename__, ver=__version__))
URL = 'https://github.com/oesteban/{}'.format(__packagename__)

REQUIRES = [
    'future',
    'numpy',
    'nipype',
    'nibabel',
    'nipy',
    'scipy',
    'phantomas',
]

LINKS_REQUIRES = [
    'git+https://github.com/oesteban/phantomas.git#egg=phantomas',
    'git+https://github.com/oesteban/nipype.git#egg=nipype',
]

TESTS_REQUIRES = ['mock', 'codecov', 'pytest-xdist']

EXTRA_REQUIRES = {
    'doc': ['sphinx'],
    'tests': TESTS_REQUIRES,
    'duecredit': ['duecredit']
}

# Enable a handle to install all extra dependencies at once
EXTRA_REQUIRES['all'] = [val for _, val in list(EXTRA_REQUIRES.items())]
