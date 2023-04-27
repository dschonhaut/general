import sys
import os.path as op
import subprocess

sys.path.append()
import general.nifti.nifti_ops as nops
import general.basic.str_methods as strm


def run_cmd(cmd):
    """Run a shell command and return the output."""
    if isinstance(cmd, str):
        cmd = cmd.split()
    output = subprocess.run(cmd, shell=True, capture_output=True)
    if output.returncode != 0:
        raise RuntimeError(output.stderr)
    return output


def gaussian_fwhm_to_sigma(fwhm):
    """Convert FWHM to sigma for Gaussian kernel."""
    sigma = fwhm / 2.3548200450309493
    return sigma


def run_fnirt(infile, templatef, prefix="r", overwrite=False):
    """Run FSL's FNIRT command line and return the outfile."""
    infile = nops.find_gzip(infile)
    templatef = nops.find_gzip(templatef)
    assert all([op.isfile(f) for f in [infile, templatef]])
    _outfile = nops.find_gzip(outfile)
    if op.isfile(_outfile) and not overwrite:
        return _outfile
    cmd = f"flirt -dof 6 -in {infile} -ref {templatef} -out {outfile}"
    output = run_cmd(cmd)
    outfile = nops.find_gzip(outfile)
    return outfile


def niimath_smooth(
    infile,
    outfile,
    fwhm=None,
    res_in=None,
    res_target=None,
    prefix="s",
    overwrite=False,
):
    """Apply 3D Gaussian smoothing to a nifti image using niimath and return the outfile."""

    cmd = "flirt -dof 6 -in s8mean_sub-052_S_1352_pet-FBP_ses-2018-10-18.nii.gz -ref /Volumes/petcore/Projects/ADNI_Reads/templates/rTemplate_FBP-all.nii -out rs8mean_sub-052_S_1352_pet-FBP_ses-2018-10-18"
