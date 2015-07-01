#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-06-23 12:32:07
# @Last Modified by:   oesteban
# @Last Modified time: 2015-07-01 08:17:45

import os
import os.path as op
import numpy as np
import nipype.pipeline.engine as pe             # pipeline engine

from nipype.interfaces import utility as niu         # utility
from nipype.interfaces import io as nio
from nipype.interfaces import fsl                    # fsl
from nipype.interfaces import freesurfer as fs       # freesurfer
from nipype.interfaces.dipy import Denoise
from nipype.interfaces import mrtrix3 as mrt3

import utils as pu
from interfaces import PhantomasSticksSim, LoadSamplingScheme, SigmoidFilter
from pyacwereg.interfaces import Surf2Vol


def gen_diffantom(name='Diffantom', settings={}):
    sgm_structures = ['L_Accu', 'R_Accu', 'L_Caud', 'R_Caud',
                      'L_Pall', 'R_Pall', 'L_Puta', 'R_Puta',
                      'L_Thal', 'R_Thal']

    def _getfirst(inlist):
        return inlist[0]

    def _sort(inlist):
        return sorted(inlist)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['subject_id', 'data_dir']), name='inputnode')
    inputnode.inputs.subject_id = settings['subject_id']
    inputnode.inputs.data_dir = settings['data_dir']

    fnames = dict(t1w='T1w_acpc_dc_restore.nii.gz',
                  fibers='fiber*.nii.gz',
                  vfractions='vfraction*.nii.gz',
                  scheme='samples.txt')

    ds_tpl_args = {k: [['subject_id', [v]]] for k, v in fnames.iteritems()}

    ds = pe.Node(nio.DataGrabber(
        infields=['subject_id'], sort_filelist=True, template='*',
        outfields=ds_tpl_args.keys()), name='DataSource')
    ds.inputs.field_template = {k: 'models/%s/%s'
                                for k in ds_tpl_args.keys()}
    ds.inputs.template_args = ds_tpl_args

    bet = pe.Node(
        fsl.BET(frac=0.15, robust=True, mask=True), name='BrainExtraction')

    fast = pe.Node(fsl.FAST(number_classes=3, img_type=1, no_bias=True,
                            probability_maps=True), name='SegmentT1')
    resfast = pe.MapNode(fs.MRIConvert(), name='ResliceFAST',
                         iterfield=['in_file'])

    first = pe.Node(fsl.FIRST(
        list_of_specific_structures=sgm_structures, brain_extracted=True,
        method='fast'), name='FIRST')
    reslice = pe.Node(fs.MRIConvert(), name='ResliceMask')
    mesh2pve = pe.MapNode(mrt3.Mesh2PVE(), iterfield=['in_file'],
                          name='Mesh2PVE')
    mfirst = pe.Node(niu.Function(
        function=pu.merge_first, input_names=['inlist'],
        output_names=['out_file']), name='FIRSTMerge')

    gen5tt = pe.Node(mrt3.Generate5tt(out_file='act5tt.nii.gz'),
                     name='Generate5TT')
    sim_mod = preprocess_model()
    sim_ref = simulate()

    fixVTK = pe.MapNode(niu.Function(
        input_names=['in_file', 'in_ref'], output_names=['out_file'],
        function=pu.fixvtk), name='fixVTK', iterfield=['in_file'])

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, ds,       [('subject_id', 'subject_id'),
                               ('data_dir', 'base_directory')]),
        (ds,        bet,      [('t1w', 'in_file')]),
        (ds,        sim_mod,  [('fibers', 'inputnode.fibers'),
                               ('vfractions', 'inputnode.vfractions')]),
        (ds,        sim_ref,  [('scheme', 'inputnode.scheme')]),
        (bet,       reslice,  [('mask_file', 'in_file')]),
        (ds,        reslice,  [(('vfractions', _getfirst), 'reslice_like')]),
        (reslice,   sim_mod,  [('out_file', 'inputnode.in_mask')]),
        (bet,       fast,     [('out_file', 'in_files')]),
        (fast,      resfast,  [(('partial_volume_files', _sort), 'in_file')]),
        (ds,        resfast,  [(('vfractions', _getfirst), 'reslice_like')]),
        (bet,       first,    [('out_file', 'in_file')]),
        (first,     fixVTK,   [('vtk_surfaces', 'in_file')]),
        (ds,        fixVTK,   [(('vfractions', _getfirst), 'in_ref')]),
        (ds,        mesh2pve, [(('vfractions', _getfirst), 'reference')]),
        (fixVTK,    mesh2pve, [('out_file', 'in_file')]),
        (mesh2pve,  mfirst,   [('out_file', 'inlist')]),
        (resfast,   gen5tt,   [('out_file', 'in_fast')]),
        (mfirst,    gen5tt,   [('out_file', 'in_first')]),
        (gen5tt,    sim_mod,  [('out_file', 'inputnode.in_5tt')]),
        (bet,       sim_ref,  [('mask_file', 'inputnode.in_mask')]),
        (sim_mod,   sim_ref,     [
            ('outputnode.fibers', 'inputnode.fibers'),
            ('outputnode.fractions', 'inputnode.fractions')]),
    ])
    return wf


def simulate(name='SimDWI'):
    in_fields = ['fibers', 'fractions', 'in_mask', 'scheme']

    inputnode = pe.Node(niu.IdentityInterface(fields=in_fields),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['dwi', 'bvec', 'bval', 'out_mask', 'out_fods']),
        name='outputnode')

    split = pe.Node(niu.Split(splits=[3, 4]), name='SplitFractions')
    sch = pe.Node(LoadSamplingScheme(bvals=[1000, 3000]), name='LoadScheme')
    simdwi = pe.Node(PhantomasSticksSim(
        lambda1=2.2e-3, lambda2=.2e-3, snr=90, save_fods=True),
        name='SimulateDWI')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, split,      [('fractions', 'inlist')]),
        (split,     simdwi,     [('out1', 'in_frac'),
                                 ('out2', 'in_5tt')]),
        (inputnode, simdwi,     [('fibers', 'in_dirs')]),
        (simdwi,    outputnode, [('out_mask', 'out_mask'),
                                 ('out_fods', 'out_fods')]),
        (inputnode, sch,        [('scheme', 'in_file')]),
        (sch,       simdwi,     [('out_bval', 'in_bval'),
                                 ('out_bvec', 'in_bvec')]),
        (sch,       outputnode, [('out_bvec', 'bvec'),
                                 ('out_bval', 'bval')])
    ])

    return wf


def preprocess_model(name='PrepareModel'):
    in_fields = ['fibers', 'vfractions', 'in_5tt', 'in_mask']

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

    spl5tt = pe.Node(fsl.Split(dimension='t'), name='Split5tt')
    sel4tt = pe.Node(niu.Select(index=range(4)), name='Select4tt')

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
        (inputnode, spl5tt,     [('in_5tt', 'in_file')]),
        (inputnode, fixtsr,     [('fibers', 'in_file')]),
        (inputnode, selvfs,     [('vfractions', 'inlist')]),
        (spl5tt,    sel4tt,     [('out_files', 'inlist')]),
        (selvfs,    mskvfs,     [('out', 'in_file')]),
        (sel4tt,    dwimsk,     [('out', 'in_file')]),
        (dwimsk,    mskvfs,     [('out_file', 'mask_file')]),
        (mskvfs,    denoise,    [('out_file', 'in_file')]),
        (inputnode, denoise,    [('in_mask', 'in_mask')]),
        (denoise,   enh,        [('out_file', 'in_file')]),
        (enh,       post,       [('out_file', 'sf_vfs')]),
        (sel4tt,    post,       [('out', 'tissue_vfs')]),
        (post,      merge,      [('out_sf', 'in1'),
                                 ('out_ts', 'in2')]),
        (merge,     outputnode, [('out', 'fractions')]),
        (fixtsr,    outputnode, [('out_file', 'fibers')]),
        (post,      outputnode, [('out_wmmsk', 'out_wmmsk')])
    ])

    return wf
