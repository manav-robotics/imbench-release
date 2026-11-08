"""Crash-safe file writes."""

from __future__ import annotations

import contextlib
import json
import os
import secrets


def atomic_write_json(
    path: str | os.PathLike,
    obj,
    *,
    indent: int = 2,
    sort_keys: bool = False,
    mode: int = 0o666,
) -> None:
    """Write *obj* to *path* as JSON so readers see either the old or the new file.

    The JSON goes to a temporary file in the same directory, which is fsynced
    and then renamed over *path*.  On any error, including KeyboardInterrupt,
    the temporary file is removed and *path* is left as it was.  *mode* is the
    permission of the new file before the process umask is applied.
    """
    path = os.fspath(path)
    directory, name = os.path.split(path)
    tmp = os.path.join(directory, f".{name}.{secrets.token_hex(4)}.tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(obj, f, indent=indent, sort_keys=sort_keys)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(tmp)
        raise
