#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-06-23 12:32:07
# @Last Modified by:   Oscar Esteban
# @Last Modified time: 2015-09-09 16:21:22
from __future__ import absolute_import
import os
import os.path as op
import numpy as np
import nipype.pipeline.engine as pe             # pipeline engine

from nipype.interfaces import utility as niu         # utility
from nipype.interfaces import io as nio
from nipype.interfaces import fsl                    # fsl
from nipype.interfaces import freesurfer as fs       # freesurfer
from nipype.interfaces import mrtrix3 as mrt3


def gen_diffantom(name='Diffantom', settings={}):

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['subject_id', 'data_dir']), name='inputnode')
    inputnode.inputs.subject_id = settings['subject_id']
    inputnode.inputs.data_dir = settings['data_dir']

    fnames = dict(t1w='T1w_acpc_dc_restore.nii.gz',
                  fibers='fiber*.nii.gz',
                  vfractions='vfraction*.nii.gz',
                  scheme='samples.txt',
                  aparc='aparc+aseg.nii.gz',
                  fa='fa.nii.gz',
                  wmparc='wmparc.nii.gz')

    ds_tpl_args = {k: [['subject_id', [v]]] for k, v in fnames.iteritems()}

    ds = pe.Node(nio.DataGrabber(
        infields=['subject_id'], sort_filelist=True, template='*',
        outfields=ds_tpl_args.keys()), name='DataSource')
    ds.inputs.field_template = {k: 'models/%s/%s'
                                for k in ds_tpl_args.keys()}
    ds.inputs.template_args = ds_tpl_args

    sim_mod = preprocess_model()
    sim_ref = simulate()
    trk = act_workflow()

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, ds,       [('subject_id', 'subject_id'),
                               ('data_dir', 'base_directory')]),
        (ds,        sim_mod,  [('t1w', 'inputnode.t1w'),
                               ('fibers', 'inputnode.fibers'),
                               ('vfractions', 'inputnode.fractions'),
                               ('aparc', 'inputnode.parcellation'),
                               ('fa', 'inputnode.in_fa')]),
        (ds,        trk,      [('aparc', 'inputnode.aparc'),
                               ('wmparc', 'inputnode.wmparc')]),
        (ds,        sim_ref,  [('scheme', 'inputnode.scheme')]),
        (sim_mod,   sim_ref,     [
            ('outputnode.fibers', 'inputnode.fibers'),
            ('outputnode.fractions', 'inputnode.fractions'),
            ('outputnode.out_mask', 'inputnode.in_mask'),
            ('outputnode.out_iso', 'inputnode.in_5tt')]),
        (sim_ref,   trk,      [
            ('outputnode.dwi', 'inputnode.in_dwi'),
            ('outputnode.out_grad', 'inputnode.in_scheme')]),
        (sim_mod,   trk,      [
            ('outputnode.out_5tt', 'inputnode.in_5tt'),
            ('outputnode.parcellation', 'inputnode.parcellation')])
    ])
    return wf


def finf_bundles(name='FINFBundles', settings={}):
    inputnode = pe.Node(niu.IdentityInterface(
        fields=['subject_id', 'data_dir']), name='inputnode')
    inputnode.inputs.subject_id = settings['subject_id']
    inputnode.inputs.data_dir = settings['data_dir']

    out_fields = ['out_file', 'out_fa', 'out_adc', 'out_tdi', 'out_map',
                  'out_cst']

    outputnode = pe.Node(
        niu.IdentityInterface(fields=out_fields), name='outputnode')

    fnames = dict(in_dwi='sim_dwi.nii.gz',
                  in_scheme='grad.txt',
                  in_5tt='act5tt.nii.gz',
                  aparc='aparc+aseg.nii.gz',
                  parcellation='aparc+aseg.nii.gz',
                  wmparc='wmparc.nii.gz')

    ds_tpl_args = {k: [['subject_id', [v]]] for k, v in fnames.iteritems()}

    ds = pe.Node(nio.DataGrabber(
        infields=['subject_id'], sort_filelist=True, template='*',
        outfields=ds_tpl_args.keys()), name='DataSource')
    ds.inputs.field_template = {k: 'subjects/%s/%s'
                                for k in ds_tpl_args.keys()}
    ds.inputs.template_args = ds_tpl_args

    trk = act_workflow()

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, ds,       [('subject_id', 'subject_id'),
                               ('data_dir', 'base_directory')]),
        (ds, trk, [(f, 'inputnode.' + f) for f in fnames.keys()]),
        (trk, outputnode, [('outputnode.' + f, f) for f in out_fields])
    ])
    return wf


def simulate(name='SimDWI'):
    from nipype.interfaces.dipy import Denoise
    from .interfaces import PhantomasSticksSim, LoadSamplingScheme

    in_fields = ['fibers', 'fractions', 'in_5tt', 'in_mask', 'scheme']

    inputnode = pe.Node(niu.IdentityInterface(fields=in_fields),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['dwi', 'bvec', 'bval', 'out_mask', 'out_fods', 'out_grad']),
        name='outputnode')

    sch = pe.Node(LoadSamplingScheme(bvals=[2000]), name='LoadScheme')
    simdwi = pe.Node(PhantomasSticksSim(
        lambda1=2.2e-3, lambda2=.2e-3, snr=30, save_fods=True),
        name='SimulateDWI')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, simdwi,     [('fibers', 'in_dirs'),
                                 ('fractions', 'in_frac'),
                                 ('in_5tt', 'in_5tt')]),
        (simdwi,    outputnode, [('out_file', 'dwi'),
                                 ('out_mask', 'out_mask'),
                                 ('out_fods', 'out_fods')]),
        (inputnode, sch,        [('scheme', 'in_file')]),
        (sch,       simdwi,     [('out_bval', 'in_bval'),
                                 ('out_bvec', 'in_bvec')]),
        (sch,       outputnode, [('out_bvec', 'bvec'),
                                 ('out_bval', 'bval'),
                                 ('out_mrtrix', 'out_grad')])
    ])

    return wf


def preprocess_model(name='PrepareModel'):
    from .interfaces import SigmoidFilter
    import diffantom.utils as pu

    in_fields = ['t1w', 'fibers', 'fractions', 'parcellation', 'in_fa']
    sgm_structures = ['L_Accu', 'R_Accu', 'L_Caud', 'R_Caud',
                      'L_Pall', 'R_Pall', 'L_Puta', 'R_Puta',
                      'L_Thal', 'R_Thal']

    def _getfirst(inlist):
        return inlist[0]

    def _sort(inlist):
        return sorted(inlist)

    inputnode = pe.Node(niu.IdentityInterface(fields=in_fields),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['fibers', 'fractions', 'out_5tt', 'out_iso', 'out_mask',
                'parcellation', 't1w_brain']), name='outputnode')

    def _getfirst(inlist):
        return inlist[0]

    selvfs = pe.Node(niu.Select(index=[1, 2, 3]), name='VFSelect')

    fixtsr = pe.MapNode(niu.Function(
        input_names=['in_file'], output_names=['out_file'],
        function=pu.fix_tensors), iterfield=['in_file'], name='FixTensors')

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

    fixVTK = pe.MapNode(niu.Function(
        input_names=['in_file', 'in_ref'], output_names=['out_file'],
        function=pu.fixvtk), name='fixVTK', iterfield=['in_file'])
    mesh2pve = pe.MapNode(mrt3.Mesh2PVE(), iterfield=['in_file'],
                          name='Mesh2PVE')
    mfirst = pe.Node(niu.Function(
        function=pu.merge_first, input_names=['inlist'],
        output_names=['out_file']), name='FIRSTMerge')
    gen5tt = pe.Node(mrt3.Generate5tt(out_file='act5tt.nii.gz'),
                     name='Generate5TT')

    spl5tt = pe.Node(fsl.Split(dimension='t'), name='Split5tt')
    sel4tt = pe.Node(niu.Select(index=range(4)), name='Select4tt')

    dwimsk = pe.Node(niu.Function(
        function=pu._sim_mask, input_names=['in_file'],
        output_names=['out_file']), name='GenSimMsk')

    mskvfs = pe.MapNode(fs.ApplyMask(), iterfield=['in_file'], name='VFMsk')

    denoise = pe.MapNode(Denoise(snr=90.0), name='VFDenoise',
                         iterfield=['in_file'])
    enh = pe.MapNode(SigmoidFilter(
        max_out=1.0, lower_perc=0.0),
        iterfield=['in_file', 'upper_perc'], name='VFEnhance')
    enh.inputs.upper_perc = [0.7, 0.85, 0.9]

    post = pe.Node(niu.Function(
        function=pu.compute_fractions, input_names=['sf_vfs', 'tissue_vfs'],
        output_names=['out_sf', 'out_ts', 'out_wmmsk']), name='VFPrepare')

    fixparc = pe.Node(mrt3.ReplaceFSwithFIRST(), name='FixFSaparc')
    fixparc.inputs.in_config = op.join(
        os.getenv('MRTRIX3_HOME', '/home/oesteban/workspace/mrtrix3'),
        'src/connectome/config/fs_default.txt')

    faden = pe.Node(Denoise(snr=90.0), name='FADenoise')
    fapst = pe.Node(niu.Function(
        function=pu.fa_fractions, input_names=['fa', 'sf_vfs', 'tissue_vfs'],
        output_names=['out_sf', 'out_ts', 'out_wmmsk']), name='FAPrepare')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, bet,        [('t1w', 'in_file')]),
        (inputnode, reslice,    [(('fractions', _getfirst), 'reslice_like')]),
        (inputnode, fixtsr,     [('fibers', 'in_file')]),
        (inputnode, selvfs,     [('fractions', 'inlist')]),
        (inputnode, resfast,    [(('fractions', _getfirst), 'reslice_like')]),
        (inputnode, fixVTK,     [(('fractions', _getfirst), 'in_ref')]),
        (inputnode, mesh2pve,   [(('fractions', _getfirst), 'reference')]),
        (inputnode, faden,      [('in_fa', 'in_file')]),
        (bet,       reslice,    [('mask_file', 'in_file')]),
        (bet,       fast,       [('out_file', 'in_files')]),
        (fast,      resfast,    [
            (('partial_volume_files', _sort), 'in_file')]),
        (bet,       first,      [('out_file', 'in_file')]),
        (first,     fixVTK,     [('vtk_surfaces', 'in_file')]),
        (fixVTK,    mesh2pve,   [('out_file', 'in_file')]),
        (mesh2pve,  mfirst,     [('out_file', 'inlist')]),
        (resfast,   gen5tt,     [('out_file', 'in_fast')]),
        (mfirst,    gen5tt,     [('out_file', 'in_first')]),
        (gen5tt,    spl5tt,     [('out_file', 'in_file')]),
        (spl5tt,    sel4tt,     [('out_files', 'inlist')]),
        (selvfs,    mskvfs,     [('out', 'in_file')]),
        (sel4tt,    dwimsk,     [('out', 'in_file')]),
        (dwimsk,    mskvfs,     [('out_file', 'mask_file')]),
        (reslice,   faden,      [('out_file', 'in_mask')]),
        (mskvfs,    denoise,    [('out_file', 'in_file')]),
        (reslice,   denoise,    [('out_file', 'in_mask')]),
        (denoise,   enh,        [('out_file', 'in_file')]),
        (enh,       post,       [('out_file', 'sf_vfs')]),
        (sel4tt,    post,       [('out', 'tissue_vfs')]),
        (sel4tt,    fapst,      [('out', 'tissue_vfs')]),
        (denoise,   fapst,      [('out_file', 'sf_vfs')]),
        (faden,     fapst,      [('out_file', 'fa')]),
        (fapst,     outputnode, [('out_sf', 'fractions'),
                                 ('out_ts', 'out_iso')]),
        (bet,       outputnode, [('out_file', 't1w_brain')]),
        (gen5tt,    outputnode, [('out_file', 'out_5tt')]),
        (fixtsr,    outputnode, [('out_file', 'fibers')]),
        (reslice,   outputnode, [('out_file', 'out_mask')]),
        (inputnode, fixparc,    [('parcellation', 'in_file'),
                                 ('t1w', 'in_t1w')]),
        (fixparc,   outputnode, [('out_file', 'parcellation')])
    ])

    return wf


def act_workflow(name='Tractography'):
    from nipype.interfaces.mrtrix import MRTrix2TrackVis as TCK2TRK
    import diffantom.utils as pu

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_dwi', 'in_scheme', 'in_5tt', 'parcellation', 'aparc',
                'wmparc']),
        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file', 'out_fa', 'out_adc', 'out_tdi', 'out_map',
                'out_cst']),
        name='outputnode')

    bmsk = pe.Node(niu.Function(
        function=pu.mask_from_5tt, input_names=['in_5tt'],
        output_names=['out_file']), name='ComputeBrainMask')

    tmsk = pe.Node(niu.Function(
        function=pu.tmask_from_5tt, input_names=['in_5tt'],
        output_names=['out_file']), name='ComputeTrackingMask')

    resp = pe.Node(mrt3.ResponseSD(
        bval_scale='yes', max_sh=8, nthreads=2), name='EstimateResponse')
    fod = pe.Node(mrt3.EstimateFOD(
        bval_scale='yes', max_sh=8, nthreads=0), name='EstimateFODs')
    tsr = pe.Node(mrt3.FitTensor(
        bval_scale='yes', nthreads=0), name='EstimateTensors')
    met = pe.Node(mrt3.TensorMetrics(
        out_adc='adc.nii.gz', out_fa='fa.nii.gz'), name='ComputeScalars')
    trk = pe.Node(mrt3.Tractography(
        nthreads=0, n_tracks=int(1e6), max_length=200.), name='Track')

    lc = pe.Node(mrt3.LabelConfig(), name='LabelConfig')
    lc.inputs.out_file = 'parcellation.nii.gz'
    lc.inputs.in_config = op.join(
        os.getenv('MRTRIX3_HOME', '/home/oesteban/workspace/mrtrix3'),
        'src/connectome/config/fs_default.txt')
    lc.inputs.lut_fs = op.join(
        os.getenv('FREESURFER_HOME'), 'FreeSurferColorLUT.txt')

    mat = pe.Node(mrt3.BuildConnectome(nthreads=0), name='BuildMatrix')

    tck2trk = pe.Node(TCK2TRK(), name='TCK2TRK')
    tck2vtk = pe.Node(mrt3.TCK2VTK(), name='TCK2VTK')

    tdi = pe.Node(mrt3.ComputeTDI(nthreads=0, out_file='tdi.nii.gz'),
                  name='ComputeTDI')

    bnd0 = track_bundle('Bundle_FMinor', incl_rois=True)
    bnd0.inputs.inputnode.seed_lbs = [254, 255,
                                      3014, 4014,
                                      3026, 4026]  # 2, 41,
    # labels = [1002, 1012, 1014, 1026, 1027, 1028, 1032]
    # labels = [1002, 1012, 1026, 1027, 1028, 1032]
    labels = [1014, 1032, 3032, 3027]
    bnd0.inputs.inputnode.roi1 = labels
    bnd0.inputs.inputnode.roi2 = (np.array(labels) + 1000).tolist()

    excl = pe.Node(niu.Function(
        function=pu.exclude_roi, input_names=['in_file'],
        output_names=['out_file']), name='ExcludeFMinor')

    bnd1 = track_bundle('Bundle_CST', incl_rois=False)
    bnd1.inputs.inputnode.seed_lbs = [16, 28, 60, 5001, 5002]
    exclCST = pe.Node(
        fs.Binarize(match=range(251, 256) + [7, 46]),
        name='ExcludeCST')

    cst = track_querier('TractQuery_CST')
    cst.inputs.inputnode.in_qry = op.join(
        os.getenv('HOME'),
        'workspace/tract_querier/doc/examples/wmql_1_cst.qry')
    fcm = track_querier('TractQuery_FCMinor')
    fcm.inputs.inputnode.in_qry = op.join(
        os.getenv('HOME'),
        'workspace/tract_querier/doc/examples/wmql_4_forcepsminor.qry')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, bmsk,       [('in_5tt', 'in_5tt')]),
        (inputnode, tmsk,       [('in_5tt', 'in_5tt')]),
        (inputnode, lc,         [('parcellation', 'in_file')]),
        (inputnode, excl,       [('aparc', 'in_file')]),
        (inputnode, tck2trk,    [('in_dwi', 'image_file')]),
        (inputnode, tck2vtk,    [('parcellation', 'reference')]),
        (inputnode, resp,       [('in_dwi', 'in_file'),
                                 ('in_scheme', 'grad_file')]),
        (bmsk,      resp,       [('out_file', 'in_mask')]),
        (bmsk,      tsr,        [('out_file', 'in_mask')]),
        (inputnode, tsr,        [('in_dwi', 'in_file'),
                                 ('in_scheme', 'grad_file')]),
        (tsr,       met,        [('out_file', 'in_file')]),
        (bmsk,      met,        [('out_file', 'in_mask')]),
        (bmsk,      fod,        [('out_file', 'in_mask')]),
        (inputnode, fod,        [('in_dwi', 'in_file'),
                                 ('in_scheme', 'grad_file')]),
        (resp,      fod,        [('out_file', 'response')]),
        (fod,       trk,        [('out_file', 'in_file')]),
        (tmsk,      trk,        [('out_file', 'seed_image')]),
        (inputnode, trk,        [('in_5tt', 'act_file'),
                                 ('in_scheme', 'grad_file')]),
        (trk,       tdi,        [('out_file', 'in_file')]),
        (bmsk,      tdi,        [('out_file', 'reference')]),
        (trk,       tck2trk,    [('out_file', 'in_file')]),
        (trk,       tck2vtk,    [('out_file', 'in_file')]),
        (inputnode, bnd0,       [
            ('wmparc', 'inputnode.wmparc'),
            ('parcellation', 'inputnode.parcellation'),
            ('in_5tt', 'inputnode.in_5tt'),
            ('in_scheme', 'inputnode.in_scheme')]),
        (excl,      bnd0,       [('out_file', 'inputnode.roi_excl')]),
        (fod,       bnd0,       [('out_file', 'inputnode.in_fod')]),

        (inputnode, bnd1,       [
            ('wmparc', 'inputnode.wmparc'),
            ('parcellation', 'inputnode.parcellation'),
            ('in_5tt', 'inputnode.in_5tt'),
            ('in_scheme', 'inputnode.in_scheme')]),
        (inputnode, exclCST,    [('wmparc', 'in_file')]),
        (exclCST,   bnd1,       [('binary_file', 'inputnode.roi_excl')]),
        (fod,       bnd1,       [('out_file', 'inputnode.in_fod')]),

        (trk,       mat,        [('out_file', 'in_file')]),
        (lc,        mat,        [('out_file', 'in_parc')]),
        (trk,       outputnode, [('out_file', 'out_file')]),
        (met,       outputnode, [('out_fa', 'out_fa'),
                                 ('out_adc', 'out_adc')]),
        (tdi,       outputnode, [('out_file', 'out_tdi')]),
        (mat,       outputnode, [('out_file', 'out_map')]),
        (inputnode, fcm,        [('wmparc', 'inputnode.in_parc')]),
        (bnd0,      fcm,        [
            ('outputnode.out_file', 'inputnode.in_track')]),
        (inputnode, cst,        [('wmparc', 'inputnode.in_parc')]),
        (bnd1,      cst,        [
            ('outputnode.out_file', 'inputnode.in_track')]),
        (cst,       outputnode, [('outputnode.out_track', 'out_cst')]),
        (fcm,       outputnode, [('outputnode.out_track', 'out_fcm')]),
        (tck2trk,   outputnode, [('out_file', 'out_trk')])
    ])
    return wf


def track_bundle(name='BundleTrack', incl_rois=True):
    from nipype.interfaces.mrtrix import MRTrix2TrackVis as TCK2TRK
    import diffantom.utils as pu

    infields = ['in_fod', 'wmparc', 'seed_lbs', 'parcellation',
                'in_5tt', 'in_scheme', 'roi_excl']

    if incl_rois:
        infields += ['roi1', 'roi2']
        inc = pe.Node(niu.Function(
            function=pu.gen_inc_rois, output_names=['out_file'],
            input_names=['in_file', 'roi1', 'roi2', 'seed_roi']),
            name='IncludeBinarize')

    msk = pe.Node(fs.Binarize(), name='SeedBinarize')
    trk = pe.Node(mrt3.Tractography(
        nthreads=0, n_tracks=int(1e5), max_length=100.), name='Track')
    tck2trk = pe.Node(TCK2TRK(), name='TCK2TRK')
    tck2vtk = pe.Node(mrt3.TCK2VTK(), name='TCK2VTK')

    # Build workflow
    inputnode = pe.Node(niu.IdentityInterface(fields=infields),
                        name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['out_file']),
        name='outputnode')
    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, tck2trk,    [('wmparc', 'image_file')]),
        (inputnode, tck2vtk,    [('wmparc', 'reference')]),
        (inputnode, msk,        [('wmparc', 'in_file'),
                                 ('seed_lbs', 'match')]),
        (inputnode, trk,        [('in_fod', 'in_file'),
                                 ('in_5tt', 'act_file'),
                                 ('in_scheme', 'grad_file'),
                                 ('roi_excl', 'roi_excl')]),
        (msk,       trk,        [('binary_file', 'seed_image')]),
        (trk,       tck2trk,    [('out_file', 'in_file')]),
        (trk,       tck2vtk,    [('out_file', 'in_file')]),
        (tck2vtk,   outputnode, [('out_file', 'out_file')])
    ])

    if incl_rois:
        wf.connect([
            (inputnode, inc,        [('wmparc', 'in_file'),
                                     ('roi1', 'roi1'),
                                     ('roi2', 'roi2')]),
            (msk,       inc,        [('binary_file', 'seed_roi')]),
            (inc,       trk,        [('out_file', 'roi_incl')])
        ])

    return wf


def track_querier(name='TractQuery'):
    from .interfaces import TractQuerier
    import diffantom.utils as pu

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['in_track', 'in_parc', 'in_qry']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(fields=['out_track']),
                         name='outputnode')

    toras = pe.Node(fs.MRIConvert(out_type='niigz', out_orientation='RAS'),
                    name='ParcToRAS')
    fixorig = pe.Node(niu.Function(
        function=pu.fixorigin, input_names=['in_file'],
        output_names=['out_file']), name='ParcOrig')

    tq = pe.Node(TractQuerier(), name='TractQuerier')

    wf = pe.Workflow(name=name)
    wf.connect([
        (inputnode, toras,      [('in_parc', 'in_file')]),
        (toras,     fixorig,    [('out_file', 'in_file')]),
        (fixorig,   tq,         [('out_file', 'in_parc')]),
        (inputnode, tq,         [('in_track', 'in_file'),
                                 ('in_qry', 'in_queries')]),
        (tq,        outputnode, [('out_file', 'out_track')])
    ])
    return wf


def gen_model(name='GenerateModel', settings={}):
    from nipype.workflows.dmri.fsl.dti import bedpostx_parallel
    wf = pe.Workflow(name=name)

    inputnode = pe.Node(niu.IdentityInterface(
        fields=['subject_id', 'data_dir']), name='inputnode')
    outputnode = pe.Node(niu.IdentityInterface(
        fields=['dyads', 'fsamples']), name='outputnode')
    inputnode.inputs.subject_id = settings['subject_id']
    inputnode.inputs.data_dir = settings['data_dir']

    fnames = dict(dwi='dwi.nii.gz',
                  mask='dwi_brainmask.nii.gz',
                  bvecs='bvecs',
                  bvals='bvals')

    ds_tpl_args = {k: [['subject_id', [v]]] for k, v in fnames.iteritems()}

    ds = pe.Node(nio.DataGrabber(
        infields=['subject_id'], sort_filelist=True, template='*',
        outfields=ds_tpl_args.keys()), name='DataSource')
    ds.inputs.field_template = {k: 'subjects/%s/%s'
                                for k in ds_tpl_args.keys()}
    ds.inputs.template_args = ds_tpl_args

    bpx = bedpostx_parallel(compute_all_outputs=False,
                            params={'n_fibres': 3, 'model': 2})
    wf.connect([
        (inputnode, ds,         [('subject_id', 'subject_id'),
                                 ('data_dir', 'base_directory')]),
        (ds,        bpx,        [(f, 'inputnode.%s' % f)
                                 for f in fnames.keys()]),
        (bpx,       outputnode, [('outputnode.dyads', 'dyads'),
                                 ('outputnode.fsamples', 'fsamples')])

    ])
    return wf
