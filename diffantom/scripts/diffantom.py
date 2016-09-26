#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: Oscar Esteban
# @Date:   2015-06-25 15:46:08
# @Last Modified by:   oesteban
# @Last Modified time: 2016-09-26 09:44:07
"""
============================
The Diffantom software layer
============================
"""
from __future__ import absolute_import

def main():
    from argparse import ArgumentParser
    from argparse import RawTextHelpFormatter
    import glob
    import os
    import os.path as op
    import sys
    from shutil import copyfileobj
    from diffantom.workflows import gen_diffantom, gen_model, finf_bundles

    parser = ArgumentParser(description='Preprocessing dMRI routine',
                            formatter_class=RawTextHelpFormatter)

    g_input = parser.add_argument_group('Input')
    g_input.add_argument('mode', choices=['evaluation', 'model', 'bundles'],
                         default='evaluation')
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

    g_input.add_argument('--nthreads', action='store', default=0,
                         type=int, help='number of repetitions')

    opts = parser.parse_args()

    settings = {}
    settings['work_dir'] = opts.work_dir
    settings['data_dir'] = op.abspath(opts.subjects_dir)
    settings['subject_id'] = opts.subject

    # Setup work_dir
    if not op.exists(opts.work_dir):
        os.makedirs(opts.work_dir)

    # Setup multiprocessing
    nthreads = opts.nthreads
    if nthreads == 0:
        from multiprocessing import cpu_count
        nthreads = cpu_count()

    settings['nthreads'] = nthreads

    plugin = 'Linear'
    plugin_args = {}
    if nthreads > 1:
        plugin = 'MultiProc'
        plugin_args = {'n_proc': nthreads, 'maxtasksperchild': 4}

    if opts.mode == 'model':
        wf = gen_model(opts.name, settings=settings)
    elif opts.mode == 'evaluation':
        wf = gen_diffantom(opts.name, settings=settings)
    elif opts.mode == 'bundles':
        wf = finf_bundles(opts.name, settings=settings)

    wf.base_dir = settings['work_dir']
    # wf.write_graph(format='pdf')
    wf.run(plugin=plugin, plugin_args=plugin_args)

if __name__ == '__main__':
    main()
