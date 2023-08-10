import sys
import os.path as op

sys.path.append(op.join(op.expanduser("~"), "code"))
import general.nifti.nifti_ops as nops
import general.basic.str_methods as strm
import general.osops.os_utils as osu


def fsl_flirt(infile, target, dof=6, prefix="r", overwrite=True, verbose=True):
    """Run FSL's FNIRT command line and return the outfile."""
    # Check that the input files exist.
    infile = nops.find_gzip(infile, raise_error=True)
    target = nops.find_gzip(target, raise_error=True)

    # Get the outfile name and check if it exists.
    outfile = strm.add_presuf(infile, prefix)
    _outfile = nops.find_gzip(outfile)
    if _outfile is not None and op.isfile(_outfile) and not overwrite:
        return _outfile

    # Run the shell command.
    cmd = f"flirt -dof {dof} -in {infile} -ref {target} -out {outfile}"
    _ = osu.run_cmd(cmd)

    # Find and return the outfile.
    outfile = nops.find_gzip(nops.gzip_nii(outfile))
    if verbose:
        print(f"  Saved {outfile}")
    return outfile


def niimath_smooth(
    infile,
    fwhm=None,
    res_in=None,
    res_target=None,
    prefix="s",
    overwrite=True,
    verbose=True,
):
    """Apply isotropic smooth using niimath and return the outfile."""

    def fwhm_to_sigma(fwhm):
        """Convert FWHM to sigma for Gaussian kernel."""
        sigma = fwhm / 2.3548200450309493
        return sigma

    # Check that the input file exists.
    infile = nops.find_gzip(infile, raise_error=True)

    # Get the outfile name and check if it exists.
    outfile = strm.add_presuf(infile, prefix=prefix)
    _outfile = nops.find_gzip(outfile)
    if _outfile is not None and op.isfile(_outfile) and not overwrite:
        return _outfile

    # Check that the smoothing kernel is valid.
    if fwhm is None:
        assert res_in is not None and res_target is not None
        for res in (res_in, res_target):
            if isinstance(res, list):
                assert len(res) == 3
                assert res[0] == res[1] == res[2]
        fwhm = nops.calc_3d_smooth(res_in, res_target, squeeze=True)
    else:
        if isinstance(fwhm, list):
            assert len(fwhm) == 3
            assert fwhm[0] == fwhm[1] == fwhm[2]
            fwhm = fwhm[0]

    # Convert FWHM to sigma.
    sigma = fwhm_to_sigma(fwhm)

    # Run the shell command.
    cmd = f"niimath {infile} -s {sigma} {outfile}"
    _ = osu.run_cmd(cmd)

    # Find and return the outfile.
    outfile = nops.find_gzip(nops.gzip_nii(outfile))
    if verbose:
        print(f"  Saved {outfile}")
    return outfile
