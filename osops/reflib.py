#!/Users/dschonhaut/mambaforge/bin/python

"""
Format reference library.
"""

import os
import os.path as op
import sys
from collections import OrderedDict as od

if __name__ == "__main__":
    if len(sys.argv) == 1:
        print(__doc__)
        exit()


def linklib(parent, check=None, parse_pdf=False, mode: str = "rename", verbose=False):
    """Link to each file at each level above it in the file hierarchy.

    Parameters
    ----------
    parent : str
        The path to the parent directory.
    check : str
        The path to the directory that checks all file names for
        duplicates with non-identical inodes.
    mode : str
        Determines behavior if a symbolic link already exists.
        "stop"
            An error is raised and the function stops running.
        "skip"
            The function skips to the next file and returns a list
            of src paths it failed to create links to.
        "rename"
            The file is renamed with a letter suffix. E.g. if the
            link file1.py exists, the code would try to create
            file1a.py. If file1a.py exists, it would try to create
            file1b.py, etc.

    Returns
    -------
    output : OrderedDict
        "renamed_files" : OrderedDict
        "failed_lookups" : list
        "duplicates" : list
        "files_renamed" : int
        "symlinks_created" : int
        "symlinks_removed" : int
    """
    if mode not in ("rename", "skip", "stop"):
        raise ValueError
    if check is not None and not op.exists(check):
        raise FileNotFoundError(check)

    # Find and remove broken symlinks.
    symlinks_removed = 0
    symlinks_removed += rm_symlinks(parent)
    if verbose and symlinks_removed > 0:
        print("Removed {} symlinks before new link creation.".format(symlinks_removed))

    # Get regular files (not dirs or symlinks).
    files = find_files_at_depth(
        parent,
        show_files=True,
        show_symlinks=False,
        show_hidden_files=False,
        keep_ext=[".pdf"],
    )
    if verbose:
        print("Found {} PDF files in {}".format(len(files), parent))

    # Create a symlink to each file in the master directory,
    # checking for and handling duplicate file names along the way.
    renamed_files = od([])
    failed_lookups = []
    duplicates = []
    files_renamed = 0
    symlinks_created = 0
    for src in files:
        srcdir, base = op.split(src)
        if check is not None:
            # Rename the paper by scraping the PDF and searching for
            # metadata in PubMed.
            # if parse_pdf:
            #     try:
            #         # ================
            #         pdfinfo = rename_paper(src, check, inplace=True, verbose=verbose)
            #         return src, pdfinfo
            #         # ================
            #         newsrc = rename_paper(src, check, inplace=True, verbose=verbose)
            #         renamed_files[src] = newsrc
            #         src = newsrc
            #         srcdir, base = op.split(src)
            #         files_renamed += 1
            #     except:
            #         failed_lookups.append(src)
            #         continue

            # Check symlink.
            dst = op.join(check, base)
            if op.exists(dst):
                if same_file(src, dst):
                    pass
                elif mode == "stop":
                    raise FileExistsError(dst)
                elif mode == "skip":
                    duplicates.append(src)
                    continue
                elif mode == "rename":
                    newsrc = get_unique_name(src, [check, srcdir])
                    os.rename(src, newsrc)
                    renamed_files[src] = newsrc
                    src = newsrc
                    dst = op.join(check, op.basename(src))
                    os.symlink(src, dst)
                    files_renamed += 1
                    symlinks_created += 1
            else:
                os.symlink(src, dst)
                symlinks_created += 1

    # Find and remove broken symlinks.
    symlinks_removed += rm_symlinks(parent)

    # Create links at each level from parent to each file's location.
    symlinks_created += fill_symlinks(parent)

    # Print runtime details.
    if verbose:
        if parse_pdf:
            print("Failed to parse {} files".format(len(failed_lookups)))
        if mode == "skip":
            print("Skipped {} duplicate files".format(len(duplicates)))
        print("Renamed {} files".format(files_renamed))
        print("Created {} symlinks".format(symlinks_created))
        print("Removed {} broken symlinks".format(symlinks_removed))

    output = od(
        [
            ("renamed_files", renamed_files),
            ("failed_lookups", failed_lookups),
            ("duplicates", duplicates),
            ("files_renamed", files_renamed),
            ("symlinks_created", symlinks_created),
            ("symlinks_removed", symlinks_removed),
        ]
    )
    return output


def alphabet_generator():
    current_letter = "a"
    while True:
        yield current_letter
        current_letter = next_letter(current_letter)


def next_letter(current_letter):
    if current_letter == "z":
        return "aa"
    elif len(current_letter) == 1:
        return chr(ord(current_letter) + 1)
    else:
        first_letter, second_letter = current_letter
        if second_letter == "z":
            return chr(ord(first_letter) + 1) + "a"
        else:
            return first_letter + chr(ord(second_letter) + 1)


def find_files_at_depth(
    parent,
    depth=None,
    show_files=True,
    show_symlinks=False,
    show_hidden_files=True,
    show_hidden_dirs=True,
    keep_ext=[],
):
    """Return all files in parent and its nested subdirs up to depth.

    Parameters
    ----------
    parent : str
        The path to the parent directory.
    depth : int, optional
        The depth of subdirectories to search.
        If None, all files in the hierarchy are found.
        If 0, only files in parent are found.
        If 1, only files in parent and its immediate subdirectories are
        found.
        Etc.
    show_files : bool, optional
        Whether to return regular files (i.e. not symlinks). The default
        value is True.
    show_symlinks : bool, optional
        Whether to return symlinks. The default value is False.
    show_hidden_files : bool, optional
        Whether to return hidden files. The default value is True.
    show_hidden_dirs : bool, optional
        Whether to search for files in hidden directories. The default
        value is True.
    keep_ext : list, optional
        A list of filename extensions. Only files with these extensions
        will be returned.

    Returns
    -------
    list
        A list of all the files found at the specified depth.
    """
    files = []

    # Walk through the directory tree and collect all files at the specified depth
    for root, dirs, filenames in os.walk(parent):
        # Skip hidden directories if show_hidden_dirs is False
        if not show_hidden_dirs:
            dirs[:] = [d for d in dirs if not d.startswith(".")]
        if depth is None or root.count(op.sep) - parent.count(op.sep) == depth:
            # Skip hidden files if hidden_files is False
            if not show_hidden_files:
                filenames = [f for f in filenames if not f.startswith(".")]
            # Skip regular files if show_files is False
            if not show_files:
                filenames = [f for f in filenames if op.islink(op.join(root, f))]
            # Skip symlinks if show_symlinks is False
            if not show_symlinks:
                filenames = [f for f in filenames if not op.islink(op.join(root, f))]
            # Keep only files with the specified extensions
            if keep_ext:
                keep_ext = [
                    ext if ext.startswith(".") else ".{}".format(ext)
                    for ext in keep_ext
                ]
                filenames = [f for f in filenames if op.splitext(f)[1] in keep_ext]
            for f in filenames:
                files.append(op.join(root, f))

    return files


def same_file(file1, file2):
    """Return whether two files are the same."""
    try:
        return os.stat(file1).st_ino == os.stat(file2).st_ino
    except FileNotFoundError:
        return None


def rm_symlinks(parent, assert_same_basename=True):
    """Recursively remove broken symlinks in parent.

    Returns the number of symlinks removed.
    """
    # Find and remove broken symlinks.
    n_removed = 0
    symlinks = find_files_at_depth(parent, show_files=False, show_symlinks=True)
    for link in symlinks:
        if op.islink(link) and not op.exists(link):
            os.remove(link)
            n_removed += 1
        elif assert_same_basename and not (
            op.basename(link) == op.basename(op.realpath(link))
        ):
            os.remove(link)
            n_removed += 1
    return n_removed


def fill_symlinks(parent):
    """Recursively link from parent down to each file's bottom dir.

    Returns the number of symlinks created.
    """
    n_created = 0
    files = find_files_at_depth(
        parent,
        show_files=True,
        show_symlinks=False,
        show_hidden_files=False,
        keep_ext=[".pdf"],
    )
    for src in files:
        srcdir, base = op.split(src)
        cwd = srcdir
        moveon = False
        while not moveon:
            moveon = cwd == parent
            dst = op.join(cwd, base)
            if not op.exists(dst):
                os.symlink(src, dst)
                n_created += 1
            cwd = op.dirname(cwd)
    return n_created


def get_unique_name(src, check):
    """Return a unique filename checking src file against check dirs."""
    srcdir = op.dirname(src)
    base = "{}.pdf".format(op.splitext(op.basename(src))[0])
    gen = alphabet_generator()
    if isinstance(check, str):
        check = [check]
    for _check in check:
        dst = op.join(_check, base)
        if op.exists(dst) and not same_file(src, dst):
            name, ext = op.splitext(base)
            while op.exists(dst):
                base = name + next(gen) + ext
                dst = op.join(_check, base)
    return op.join(srcdir, base)


if __name__ == "__main__":
    lib = op.normpath(sys.argv[1])
    if not op.exists(lib):
        raise FileNotFoundError(f"Cannot locate path to library {lib}")
    check = op.join(lib, ".all")
    parse_pdf = False
    mode = "rename"
    verbose = True
    output = linklib(lib, check=check, parse_pdf=parse_pdf, mode=mode, verbose=verbose)
