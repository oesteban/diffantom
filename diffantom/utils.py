#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Author: oesteban
# @Date:   2015-06-23 12:32:16
# @Last Modified by:   Oscar Esteban
# @Last Modified time: 2015-06-25 20:49:48


def _sim_mask(in_file):
    import nibabel as nb
    import numpy as np
    import os.path as op
    import scipy.ndimage as sn
    from scipy.ndimage.morphology import (binary_opening, binary_dilation)
    from pyacwereg.misc import ball as gen_ball

    if not isinstance(in_file, basestring):
        in_file = in_file[2]

    out_file = op.abspath('sim_mask.nii.gz')

    im = nb.load(in_file)
    data = im.get_data()
    data[data > 1.0e-4] = 1
    data[data < 1] = 0

    ball1 = gen_ball(4, 1.9)
    data = binary_opening(data.astype(np.uint8), structure=ball1,
                          iterations=1).astype(np.uint8)
    # Get largest object
    label_im, nb_labels = sn.label(data)
    sizes = sn.sum(data, label_im, range(nb_labels + 1))
    larger = np.squeeze(np.argwhere(sizes == sizes.max()))
    data[label_im != larger] = 0

    # Dilate
    # data = binary_dilation(data, structure=ball1,
    #                        iterations=1).astype(np.uint8)

    nb.Nifti1Image(
        data, im.get_affine(), im.get_header()).to_filename(out_file)
    return out_file


def fix_tensors(in_file):
    import os.path as op
    import nibabel as nb
    import numpy as np
    out_file = op.abspath('tensors_fixed.nii.gz')
    im = nb.load(in_file)
    data = im.get_data()
    data[..., 0] *= -1.0
    # data[..., 2] *= -1.0

    nb.Nifti1Image(
        data, im.get_affine(), im.get_header()).to_filename(out_file)
    return out_file


def compute_fractions(sf_vfs, tissue_vfs, max_fa=0.80):
    import os.path as op
    import nibabel as nb
    import numpy as np
    import scipy.ndimage as sn
    from pyacwereg.misc import ball as gen_ball
    from scipy.ndimage.morphology import binary_erosion

    # Load tissue fractions
    ntissues = len(tissue_vfs)
    timgs = [nb.load(f) for f in tissue_vfs]
    tvfs = np.nan_to_num(nb.concat_images(timgs).get_data())
    tissue_totals = np.sum(tvfs, axis=3)
    tvfs[tissue_totals > 0, ...] /= tissue_totals[tissue_totals > 0,
                                                  np.newaxis]
    wmvf = tvfs[..., 2].copy()
    tvfs[..., 2] *= .05

    # Load single-fiber fractions
    nfibers = len(sf_vfs)
    sfimgs = [nb.load(f) for f in sf_vfs]
    data = np.nan_to_num(nb.concat_images(sfimgs).get_data())
    data = np.sort(data)[..., ::-1]
    data *= np.array([5, 3, 2])[np.newaxis, np.newaxis, np.newaxis, ...]
    total_fibers = np.sum(data, axis=3)
    data[total_fibers > 0, ...] /= total_fibers[total_fibers > 0, np.newaxis]
    data *= wmvf[..., np.newaxis] * .95
    total_fibers = np.sum(data, axis=3)
    tvfs[..., 2] += wmvf - total_fibers
    tvfs = np.clip(tvfs, 0.0, 1.0)

    # Save volume fractions
    out_ts = [op.abspath('ts_vf%02d.nii.gz' % i) for i in range(ntissues)]
    for i, f in enumerate(out_ts):
        nb.Nifti1Image(tvfs[..., i], timgs[i].get_affine(),
                       timgs[i].get_header()).to_filename(f)

    out_sf = [op.abspath('sf_vf%02d.nii.gz' % i) for i in range(nfibers)]
    for i, f in enumerate(out_sf):
        nb.Nifti1Image(data[..., i], sfimgs[i].get_affine(),
                       sfimgs[i].get_header()).to_filename(f)

    # Compute single fiber mask
    wmmask = np.zeros_like(total_fibers)
    wmmask[total_fibers > 0.85] = 1
    wmmask = binary_erosion(wmmask, structure=gen_ball(5, 2.4),
                            iterations=1).astype(np.uint8)

    # Get largest connected object
    label_im, nb_labels = sn.label(wmmask)
    sizes = sn.sum(wmmask, label_im, range(nb_labels + 1))
    larger = np.squeeze(np.argwhere(sizes == sizes.max()))
    wmmask[label_im != larger] = 0

    hdr = sfimgs[0].get_header().copy()
    hdr.set_data_dtype(np.uint8)

    out_wmmsk = op.abspath('wmmsk.nii.gz')
    nb.Nifti1Image(wmmask, sfimgs[0].get_affine(), hdr).to_filename(out_wmmsk)

    return out_sf, out_ts, out_wmmsk


def sigmoid_filter(data, mask=None, a=2.00, b=85.0, maxout=2000.0):
    import numpy as np

    msk = np.zeros_like(data)
    if mask is None:
        msk[data > 0] = 1.0
        data[data <= 0] = 0.0
    else:
        msk[mask > 0] = 1.0

    d = np.ma.masked_array(data.astype(np.float32), mask=1 - msk)
    maxi = d.max()
    mini = d.min()

    idxs = np.where(msk > 0)
    umdata = data[idxs]

    if maxout is None or maxout == 0.0:
        maxout = np.percentile(umdata, 99.8)

    alpha = np.percentile(umdata, a)
    beta = np.percentile(umdata, b)
    A = 2.0 / (beta - alpha)
    B = (alpha + beta) / (beta - alpha)
    res = A * umdata - B
    ddor = 1.0 / (np.exp(-res) + 1.0)
    offset = ddor[ddor > 0].min()
    ddor = ddor - offset
    newmax = ddor[ddor > 0].max()
    ddor = ddor * maxout / newmax
    newdata = np.zeros_like(data)
    newdata[idxs] = ddor
    return newdata


def fixvtk(in_file, in_ref, out_file=None):
    """
    Translates surfaces to the corresponding origin
    """
    import nibabel as nb
    import numpy as np
    import os.path as op
    import subprocess as sp

    if out_file is None:
        fname, ext = op.splitext(op.basename(in_file))
        if ext == ".gz":
            fname, _ = op.splitext(fname)
        out_file = op.abspath("%s_fixed.vtk" % fname)

    im = nb.load(in_ref)
    aff = im.get_affine()
    mni_orig = aff[:3, 3]
    flip = np.ones(3, dtype=np.float32)
    flip[mni_orig > 0.0] = -1.0

    with open(in_file, 'r') as f:
        with open(out_file, 'w+') as w:
            npoints = 0
            pointid = -5

            w.write("# vtk DataFile Version 1.0\n"
                    "vtk output\n"
                    "ASCII\n"
                    "DATASET POLYDATA\n")

            for i, l in enumerate(f):
                if (i == 4):
                    s = l.split()
                    npoints = int(s[1])
                    fmt = np.dtype(s[2])
                    w.write(l)
                elif ((i > 4) and (pointid < npoints)):
                    vert = np.array([float(x) for x in l.split()])
                    newvert = vert * flip + mni_orig
                    l = '%.5f  %.5f  %.5f\n' % tuple(newvert)
                    w.write(l)
                elif pointid == npoints:
                    s = l.split()
                    w.write("POLYGONS %d %d\n" % (int(s[1]), int(s[2])))
                elif pointid > npoints:
                    w.write(l)

                pointid += 1

    return out_file
