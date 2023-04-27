import subprocess


def run_cmd(cmd):
    """Run a shell command and return the output."""
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    output = subprocess.run(cmd, shell=True, capture_output=True)
    if output.returncode != 0:
        raise RuntimeError(output.stderr)
    return output
