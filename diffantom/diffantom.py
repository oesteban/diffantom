#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Oscar Esteban
# @Date:   2015-06-25 15:46:08
# @Last Modified by:   Oscar Esteban
# @Last Modified time: 2015-06-25 16:51:04

"""
============================
The Diffantom software layer
============================
"""
import os
import os.path as op
import sys


__author__ = "Oscar Esteban"
__copyright__ = "Copyright 2015, Biomedical Image Technologies (BIT), \
                 Universidad Politécnica de Madrid"
__credits__ = "Oscar Esteban"
__license__ = "MIT"
__version__ = "0.1"
__maintainer__ = "Oscar Esteban"
__email__ = "code@oscaresteban.es"
__status__ = "Prototype"


if __name__ == '__main__':
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter
    from workflows import gen_diffantom
    import glob
    import os
    import os.path as op
    from shutil import copyfileobj

    parser = ArgumentParser(description='Preprocessing dMRI routine',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Input')
    g_input.add_argument('-S', '--subjects_dir', action='store',
                         default=os.getenv('NEURO_DATA_HOME',
                                           op.abspath('../')),
                         help='directory where subjects should be found')
    g_input.add_argument('-s', '--subject', action='store',
                         default='S*', help='subject id or pattern')
    g_input.add_argument('-w', '--work_dir', action='store',
                         default=os.getcwd(),
                         help='directory to store intermediate results')

    g_input.add_argument(
        '-N', '--name', action='store', default='Diffantom',
        help=('default workflow name, it will create a new folder'))

    opts = parser.parse_args()

    settings = {}
    settings['work_dir'] = opts.work_dir
    settings['data_dir'] = opts.subjects_dir
    settings['subject_id'] = opts.subject

    wf = gen_diffantom(opts.name, settings=settings)

    wf.base_dir = settings['work_dir']
    # wf.write_graph(format='pdf')
    wf.run()
