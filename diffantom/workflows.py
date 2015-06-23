#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-06-23 12:32:07
# @Last Modified by:   oesteban
# @Last Modified time: 2015-06-23 12:37:25

import os
import os.path as op
import numpy as np
import nipype.pipeline.engine as pe             # pipeline engine

from nipype.interfaces import utility as niu         # utility
from nipype.interfaces import fsl                    # fsl
from nipype.interfaces import freesurfer as fs       # freesurfer
from nipype.interfaces.dipy import Denoise

import utils as pu
from interfaces import PhantomasSticksSim, LoadSamplingScheme, SigmoidFilter


def simulate(name='SimDWI', icorr=True, btable=False):
    in_fields = ['fibers', 'fractions', 'in_mask']
    if icorr:
        in_fields += ['jacobian']

    if btable:
        in_fields += ['bval', 'bvec']
    else:
        in_fields += ['scheme']

    inputnode = pe.Node(niu.IdentityInterface(fields=in_fields),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['dwi', 'bvec', 'bval', 'out_mask', 'out_fods']),
        name='outputnode')

    split = pe.Node(niu.Split(splits=[3, 3]), name='SplitFractions')
    sch = pe.Node(LoadSamplingScheme(bvals=[1000, 3000]), name='LoadScheme')
    simdwi = pe.Node(PhantomasSticksSim(
        diff_gm=1.0e-3, diff_wm=0.9e-3, diff_csf=3.0e-3,
        lambda1=2.2e-3, lambda2=.2e-3, snr=90, save_fods=True),
        name='SimulateDWI')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, split,      [('fractions', 'inlist')]),
        (split,     simdwi,     [('out1', 'in_frac'),
                                 ('out2', 'in_vfms')]),
        (inputnode, simdwi,     [('fibers', 'in_dirs')]),
        (simdwi,    outputnode, [('out_mask', 'out_mask'),
                                 ('out_fods', 'out_fods')])
    ])

    if icorr:
        split = pe.Node(fsl.utils.Split(dimension='t'), name='Split_DWIs')
        merge = pe.Node(fsl.utils.Merge(dimension='t'), name='Merge_DWIs')

        jacmask = pe.Node(fs.ApplyMask(), name='JacobianMask')
        jacmult = pe.MapNode(fsl.MultiImageMaths(op_string='-mul %s'),
                             iterfield=['in_file'], name='ModulateDWIs')
        wf.connect([
            (inputnode,     jacmask, [('jacobian', 'in_file'),
                                      ('in_mask', 'mask_file')]),
            (jacmask,       jacmult, [('out_file', 'operand_files')]),
            (simdwi,          split, [('out_file', 'in_file')]),
            (split,         jacmult, [('out_files', 'in_file')]),
            (jacmult,         merge, [('out_file', 'in_files')]),
            (merge,      outputnode, [('merged_file', 'dwi')])
        ])
    else:
        wf.connect(simdwi, 'out_file', outputnode, 'dwi')

    if btable:
        wf.connect([
            (inputnode, simdwi, [('bval', 'in_bval'), ('bvec', 'in_bvec')]),
            (inputnode, outputnode, [('bvec', 'bvec'),
                                     ('bval', 'bval')])
        ])
    else:
        wf.connect([
            (inputnode, sch,        [('scheme', 'in_file')]),
            (sch,       simdwi,     [('out_bval', 'in_bval'),
                                     ('out_bvec', 'in_bvec')]),
            (sch,       outputnode, [('out_bvec', 'bvec'),
                                     ('out_bval', 'bval')])
        ])

    return wf


def preprocess_model(name='PrepareModel'):
    in_fields = ['fibers', 'vfractions', 'in_tpms', 'in_mask']

    inputnode = pe.Node(niu.IdentityInterface(fields=in_fields),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fibers', 'fractions', 'out_wmmsk']),
        name='outputnode')

    def _getfirst(inlist):
        return inlist[0]

    selvfs = pe.Node(niu.Select(index=[1, 2, 3]), name='VFSelect')

    fixtsr = pe.MapNode(niu.Function(
        input_names=['in_file'], output_names=['out_file'],
        function=pu.fix_tensors), iterfield=['in_file'], name='FixTensors')

    regtpm = pe.MapNode(fs.MRIConvert(out_type='niigz', out_datatype='float'),
                        name='RegridTPMs', iterfield=['in_file'])

    dwimsk = pe.Node(niu.Function(
        function=pu._sim_mask, input_names=['in_file'],
        output_names=['out_file']), name='GenSimMsk')

    mskvfs = pe.MapNode(fs.ApplyMask(), iterfield=['in_file'], name='VFMsk')

    denoise = pe.MapNode(Denoise(snr=100.0), name='VFDenoise',
                         iterfield=['in_file'])
    enh = pe.MapNode(SigmoidFilter(
        max_out=1.0, lower_perc=0.0),
        iterfield=['in_file', 'upper_perc'], name='VFEnhance')
    enh.inputs.upper_perc = [0.5, 0.7, 0.8]

    post = pe.Node(niu.Function(
        function=pu.compute_fractions, input_names=['sf_vfs', 'tissue_vfs'],
        output_names=['out_sf', 'out_ts', 'out_wmmsk']), name='VFPrepare')

    merge = pe.Node(niu.Merge(2), name='VFMerge')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, regtpm,     [('in_tpms', 'in_file'),
                                 (('vfractions', _getfirst), 'reslice_like')]),
        (inputnode, fixtsr,     [('fibers', 'in_file')]),
        (inputnode, selvfs,     [('vfractions', 'inlist')]),
        (selvfs,    mskvfs,     [('out', 'in_file')]),
        (regtpm,    dwimsk,     [('out_file', 'in_file')]),
        (dwimsk,    mskvfs,     [('out_file', 'mask_file')]),
        (mskvfs,    denoise,    [('out_file', 'in_file')]),
        (inputnode, denoise,    [('in_mask', 'in_mask')]),
        (denoise,   enh,        [('out_file', 'in_file')]),
        (enh,       post,       [('out_file', 'sf_vfs')]),
        (regtpm,    post,       [('out_file', 'tissue_vfs')]),
        (post,      merge,      [('out_sf', 'in1'),
                                 ('out_ts', 'in2')]),
        (merge,     outputnode, [('out', 'fractions')]),
        (fixtsr,    outputnode, [('out_file', 'fibers')]),
        (post,      outputnode, [('out_wmmsk', 'out_wmmsk')])
    ])

    return wf
