import subprocess


def run_cmd(cmd):
    """Run a shell command and return the output."""
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True)
    if output.returncode != 0:
        raise RuntimeError(output.stderr)
    return output


def rm_files(topdir, rm_topdir=False):
    """Remove all files and directories in a directory."""
    if rm_topdir:
        output = run_cmd(f"find {topdir} -delete")
    else:
        output = run_cmd(f"find {topdir} -mindepth 1 -delete")
    return output
