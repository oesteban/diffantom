#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-06-23 12:29:58
# @Last Modified by:   oesteban
# @Last Modified time: 2015-07-13 11:47:13
from __future__ import absolute_import
import os
import os.path as op
import glob
import warnings
import numpy as np
import nibabel as nb
from math import exp
from nipype.interfaces.base import (TraitedSpec, File, InputMultiPath,
                                    OutputMultiPath, Undefined, traits,
                                    isdefined,
                                    CommandLineInputSpec, CommandLine,
                                    BaseInterface, BaseInterfaceInputSpec,
                                    traits)
from nipype.utils.filemanip import split_filename, fname_presuffix
from nipype import logging

from .utils import sigmoid_filter

iflogger = logging.getLogger('interface')


class PhantomasSticksSimInputSpec(CommandLineInputSpec):
    in_dirs = InputMultiPath(
        File(exists=True), mandatory=True, argstr='--sticks_dir %s',
        desc='list of fibers (principal directions)')
    in_frac = InputMultiPath(
        File(exists=True), mandatory=True, argstr='--sticks_vfs %s',
        desc=('volume fraction of each fiber'))
    in_5tt = InputMultiPath(
        File(exists=True), mandatory=True, argstr='--tissue_vf %s',
        desc=('volume fractions of isotropic compartiments in 5TT format'))

    in_bvec = File(exists=True, argstr='-r %s',
                   mandatory=True, desc='input bvecs file')
    in_bval = File(exists=True, argstr='-b %s',
                   mandatory=True, desc='input bvals file')

    snr = traits.Int(100, argstr='--snr %f', usedefault=True,
                     desc='signal-to-noise ratio (dB)')

    diff_5tt = traits.Tuple(
        (7e-4, 9e-4, 2e-4, 3.e-3, 1e-4), traits.Float(), traits.Float(),
        traits.Float(), traits.Float(), traits.Float(), usedefault=True,
        argstr='--diff_iso %f %f %f %f %f',
        desc='diffusion of isotropic compartments')

    lambda1 = traits.Float(2.2e-3, usedefault=True, argstr='--lambda1 %f',
                           desc='First eigenvalue of simulated tensor')
    lambda2 = traits.Float(.2e-3, usedefault=True, argstr='--lambda2 %f',
                           desc=('Second and third eigenvalues of '
                                 'simulated tensor'))

    n_proc = traits.Int(0, usedefault=True, desc='number of processes')

    out_file = File('sim_dwi.nii.gz', usedefault=True, argstr='--output %s',
                    desc='output file with fractions to be simluated')
    out_mask = File('sim_msk.nii.gz', usedefault=True, argstr='--out_mask %s',
                    desc='file with the mask simulated')

    save_fods = traits.Bool(
        False, usedefault=True, argstr='--export_fod mrtrix',
        desc='Save FODs in MRTrix format')


class PhantomasSticksSimOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='simulated DWIs')
    out_mask = File(exists=True, desc='mask file')
    out_fods = File(exists=True, desc='FODs file')


class PhantomasSticksSim(CommandLine):

    """
    Use phantomas to simulate signal from sticks


    Example
    -------
    >>> from pysdcev.interfaces.simulation import PhantomasSticksSim
    >>> sim = PhantomasSticksSim()
    >>> sim.inputs.in_dirs = ['fdir00.nii', 'fdir01.nii']
    >>> sim.inputs.in_frac = ['ffra00.nii', 'ffra01.nii']
    >>> sim.inputs.in_vfms = ['tpm_00.nii.gz', 'tpm_01.nii.gz',
    ...                       'tpm_02.nii.gz']
    >>> sim.inputs.baseline = 'b0.nii'
    >>> sim.inputs.in_bvec = 'bvecs'
    >>> sim.inputs.in_bval = 'bvals'
    >>> sim.cmdline
    'sticks2dwis '
    """
    input_spec = PhantomasSticksSimInputSpec
    output_spec = PhantomasSticksSimOutputSpec
    _cmd = 'sticks2dwis'

    def _list_outputs(self):
        outputs = self._outputs().get()
        outputs['out_file'] = op.abspath(self.inputs.out_file)
        outputs['out_mask'] = op.abspath(self.inputs.out_mask)

        if self.inputs.save_fods:
            outputs['out_fods'] = op.abspath('fods.nii.gz')

        return outputs


class SigmoidFilterInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True,
                   desc='input image')
    in_mask = File(exists=True, desc='binary mask')
    max_out = traits.Float(2000.0, mandatory=True, usedefault=True,
                           desc='fit maximum value of output')
    upper_perc = traits.Float(92.0, mandatory=True, usedefault=True,
                              desc='upper percentile for computations')
    lower_perc = traits.Float(2.0, mandatory=True, usedefault=True,
                              desc='lower percentile for computations')
    out_file = File(desc='enhanced image')


class SigmoidFilterOutputSpec(TraitedSpec):
    out_file = File(exists=True, desc='enhanced image')


class SigmoidFilter(BaseInterface):

    """
    An enhancement filter for MI-based registrations

    Example
    -------

    >>> from pysdcev.interfaces.misc import SigmoidFilter
    >>> enh = SigmoidFilter()
    >>> enh.inputs.in_file = 'T2.nii.gz'
    >>> result = enh.run() # doctest: +SKIP
    """
    input_spec = SigmoidFilterInputSpec
    output_spec = SigmoidFilterOutputSpec

    def _run_interface(self, runtime):
        im = nb.load(self.inputs.in_file)
        msk = None

        if isdefined(self.inputs.in_mask):
            msk = nb.load(self.inputs.in_mask).get_data()

        lower = self.inputs.lower_perc
        upper = self.inputs.upper_perc
        maxout = self.inputs.max_out

        enhanced = sigmoid_filter(im.get_data(), msk,
                                  a=lower, b=upper, maxout=maxout)
        outputs = self._list_outputs()

        nb.Nifti1Image(enhanced, im.get_affine(),
                       im.get_header()).to_filename(outputs['out_file'])
        return runtime

    def _list_outputs(self):
        fname, fext = op.splitext(op.basename(self.inputs.in_file))
        if fext == '.gz':
            fname, fext2 = op.splitext(fname)
            fext = fext2 + fext

        out_file = None
        if not isdefined(self.inputs.out_file):
            out_file = op.abspath(fname) + '_enh' + fext
        else:
            out_file = self.inputs.out_file

        outputs = self._outputs().get()
        outputs['out_file'] = out_file
        return outputs


class LoadSamplingSchemeInputSpec(BaseInterfaceInputSpec):
    in_file = File(exists=True, mandatory=True, desc='sampling scheme file')
    bvals = traits.List(traits.Float(), desc='bvalues')
    n_b0 = traits.Int(1, usedefault=True, desc='number of b0')
    out_bval = File('scheme.bval', usedefault=True, desc='output bvals')
    out_bvec = File('scheme.bvec', usedefault=True, desc='output bvals')
    out_mrtrix = File('grad.txt', usedefault=True, desc='output bvals')


class LoadSamplingSchemeOutputSpec(TraitedSpec):
    out_bval = File(desc='output bvals')
    out_bvec = File(desc='output bvecs')
    out_fsl = OutputMultiPath(desc='output b-matrix in fsl format')
    out_mrtrix = File(desc='output sampling scheme for use in MRTrix')


class LoadSamplingScheme(BaseInterface):

    """
    Read a scheme file generated with
    http://www.emmanuelcaruyer.com/q-space-sampling.php
    """
    input_spec = LoadSamplingSchemeInputSpec
    output_spec = LoadSamplingSchemeOutputSpec

    def _run_interface(self, runtime):
        sch = np.loadtxt(self.inputs.in_file)

        bvals = np.squeeze(sch[:, 0])
        bvecs = sch[:, 1:]

        if isdefined(self.inputs.bvals):
            if len(self.inputs.bvals) != bvals.max():
                raise RuntimeError(('Provided b-values do not match '
                                    'number of shells of the scheme'))
            for i, b in enumerate(self.inputs.bvals):
                bvals[bvals == i + 1] = b
        else:
            bvals *= 1000

        total = len(bvals) + self.inputs.n_b0
        if self.inputs.n_b0 > 0:
            sp = total // self.inputs.n_b0
            for i in range(self.inputs.n_b0):
                bvals = np.insert(bvals, i * sp, 0.0, axis=0)
                bvecs = np.insert(bvecs, i * sp, np.zeros(3), axis=0)

        np.savetxt(op.abspath(self.inputs.out_mrtrix),
                   np.vstack((bvecs.T, bvals)).T, fmt='%.4f')

        np.savetxt(op.abspath(self.inputs.out_bvec),
                   bvecs.T, fmt='%.6f')

        np.savetxt(op.abspath(self.inputs.out_bval),
                   bvals, fmt='%.2f')

        return runtime

    def _list_outputs(self):
        outputs = self._outputs().get()

        outputs['out_mrtrix'] = op.abspath(self.inputs.out_mrtrix)
        outputs['out_fsl'] = [op.abspath(self.inputs.out_bvec),
                              op.abspath(self.inputs.out_bval)]
        outputs['out_bvec'] = outputs['out_fsl'][0]
        outputs['out_bval'] = outputs['out_fsl'][1]
        return outputs


class TractQuerierInputSpec(CommandLineInputSpec):
    in_file = File(exists=True, argstr='-t %s', mandatory=True,
                   desc='input tracks in VTK format')
    in_parc = File(exists=True, argstr='-a %s', mandatory=True,
                   desc='input parcellation file')
    in_queries = File(exists=True, argstr='-q %s', mandatory=True,
                      desc='input query file')

    out_prefix = traits.Str(
        'query', argstr='-o %s', mandatory=True, usedefault=True,
        desc='output prefix')


class TractQuerierOutputSpec(TraitedSpec):
    out_file = OutputMultiPath(File(exists=True),
                               desc='the output response file')


class TractQuerier(CommandLine):

    """

    """

    _cmd = 'tract_querier'
    input_spec = TractQuerierInputSpec
    output_spec = TractQuerierOutputSpec

    def _list_outputs(self):
        outputs = self.output_spec().get()
        outputs['out_file'] = [
            op.abspath(f) for f in glob.glob('%s*' % self.inputs.out_prefix)]
        return outputs
