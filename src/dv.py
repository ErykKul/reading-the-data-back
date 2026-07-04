"""
dv.py
-----
Thin, stdlib-only helpers over the Harvard Dataverse Native API.

Public surface
--------------
    search(q, per_page=20, server=DATAVERSE_SERVER) -> list[dict]
        Full-text search for datasets.  Returns the ``data.items`` list from
        the API response (each item has at minimum ``name``, ``global_id``,
        ``url``, ``description``).

    list_files(pid, server=DATAVERSE_SERVER) -> list[dict]
        List data files in a dataset by persistent identifier (DOI).
        Returns a list of file metadata dicts; each has at minimum
        ``dataFile.id``, ``dataFile.filename``, ``dataFile.contentType``,
        ``dataFile.filesize``.

    fetch_file(file_id, dest=None, server=DATAVERSE_SERVER) -> bytes | pathlib.Path
        Download a file by Dataverse numeric file ID.
        If ``dest`` is given (a path), write the content there and return the
        Path.  Otherwise return the raw bytes.

All functions accept an optional ``server`` keyword so you can point at a
different Dataverse instance.  Authentication via the DATAVERSE_API_KEY
environment variable (optional; many datasets are public).

Environment variables
---------------------
    DATAVERSE_API_KEY   Optional Dataverse API token (for restricted files).
    DATAVERSE_SERVER    Override the default server (default: Harvard).

Example
-------
    from dv import search, list_files, fetch_file

    hits = search("hate crimes count OLS replication")
    pid  = hits[0]["global_id"]          # e.g. "doi:10.7910/DVN/HRK5HI"
    files = list_files(pid)
    for f in files:
        print(f["dataFile"]["filename"], f["dataFile"]["contentType"])
    data = fetch_file(files[0]["dataFile"]["id"])
"""

import json
import os
import pathlib
import urllib.error
import urllib.parse
import urllib.request
from typing import Optional, Union

# ── Defaults ─────────────────────────────────────────────────────────────────

DATAVERSE_SERVER: str = os.environ.get(
    "DATAVERSE_SERVER", "https://dataverse.harvard.edu"
).rstrip("/")

_DEFAULT_TIMEOUT: int = 60   # seconds


def _api_key() -> Optional[str]:
    """Return the API key from the environment, or None."""
    return os.environ.get("DATAVERSE_API_KEY") or None


def _build_headers() -> dict:
    headers = {
        "Accept": "application/json",
        # Harvard Dataverse 403s the default urllib User-Agent; a descriptive UA is accepted.
        "User-Agent": "Mozilla/5.0 (research; rdm-integration; contact eryk.kulikowski@kuleuven.be)",
    }
    key = _api_key()
    if key:
        headers["X-Dataverse-key"] = key
    return headers


def _get_json(url: str, timeout: int = _DEFAULT_TIMEOUT) -> dict:
    """
    Perform an HTTP GET on ``url`` and return the parsed JSON body.
    Raises urllib.error.HTTPError / urllib.error.URLError on failure.
    """
    req = urllib.request.Request(url, headers=_build_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read()
    return json.loads(body)


def _get_bytes(url: str, timeout: int = _DEFAULT_TIMEOUT) -> bytes:
    """Perform an HTTP GET and return raw bytes (for file download)."""
    req = urllib.request.Request(url, headers=_build_headers())
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


# ── Public API ────────────────────────────────────────────────────────────────

def search(
    q: str,
    per_page: int = 20,
    server: str = DATAVERSE_SERVER,
) -> list[dict]:
    """
    Search Harvard Dataverse (or another Dataverse instance) for datasets.

    Parameters
    ----------
    q : str
        Free-text query string (same syntax as the Dataverse search box).
    per_page : int
        Number of results to return (max 1000 per API page).
    server : str
        Base URL of the Dataverse instance.

    Returns
    -------
    list of dicts
        Each dict is a Dataverse search ``item`` object with at minimum:
        ``name``, ``global_id``, ``url``, ``description``, ``authors``,
        ``published_at``, ``subjects``.

    Raises
    ------
    urllib.error.URLError / HTTPError on network or API failure.
    ValueError if the API response is not in the expected shape.

    Example
    -------
    >>> hits = search("hate crimes count OLS replication", per_page=5)
    >>> hits[0]["global_id"]
    'doi:10.7910/DVN/HRK5HI'
    """
    params = urllib.parse.urlencode({
        "q": q,
        "type": "dataset",
        "per_page": per_page,
    })
    url = f"{server.rstrip('/')}/api/search?{params}"
    body = _get_json(url)
    status = body.get("status", "")
    if status != "OK":
        raise ValueError(f"Dataverse search returned non-OK status: {status!r}. Body: {body}")
    items = body.get("data", {}).get("items", [])
    return items


def list_files(
    pid: str,
    server: str = DATAVERSE_SERVER,
) -> list[dict]:
    """
    List the data files in a dataset identified by its persistent identifier
    (DOI or handle).

    Parameters
    ----------
    pid : str
        Persistent identifier, e.g. ``"doi:10.7910/DVN/HRK5HI"`` or
        ``"10.7910/DVN/HRK5HI"`` (the ``doi:`` prefix is added if absent).
    server : str
        Base URL of the Dataverse instance.

    Returns
    -------
    list of dicts
        Each dict is a ``latestVersion.files[]`` entry containing a nested
        ``dataFile`` dict with: ``id``, ``filename``, ``contentType``,
        ``filesize``, ``md5``, ``description``.

    Raises
    ------
    urllib.error.URLError / HTTPError on network or API failure.
    ValueError if the API response is unexpected.

    Example
    -------
    >>> files = list_files("doi:10.7910/DVN/HRK5HI")
    >>> [(f["dataFile"]["filename"], f["dataFile"]["id"]) for f in files[:3]]
    """
    # Normalise the PID: the API needs the doi: prefix
    pid = pid.strip()
    if not pid.startswith("doi:") and not pid.startswith("hdl:"):
        pid = "doi:" + pid

    encoded_pid = urllib.parse.quote(pid, safe=":/.")
    url = f"{server.rstrip('/')}/api/datasets/:persistentId/?persistentId={encoded_pid}"
    body = _get_json(url)
    status = body.get("status", "")
    if status != "OK":
        raise ValueError(
            f"list_files: API returned status {status!r} for pid={pid!r}. Body: {body}"
        )
    files = body.get("data", {}).get("latestVersion", {}).get("files", [])
    return files


def fetch_file(
    file_id: Union[int, str],
    dest: Optional[Union[str, pathlib.Path]] = None,
    server: str = DATAVERSE_SERVER,
) -> Union[bytes, pathlib.Path]:
    """
    Download a Dataverse file by its numeric file ID.

    Parameters
    ----------
    file_id : int | str
        The numeric Dataverse file ID (from ``list_files()[i]["dataFile"]["id"]``).
    dest : str | pathlib.Path | None
        If given, write the downloaded bytes to this path and return the Path.
        If None, return the raw bytes (suitable for in-memory processing or
        wrapping in io.BytesIO / io.TextIOWrapper).
    server : str
        Base URL of the Dataverse instance.

    Returns
    -------
    bytes
        Raw file content, if ``dest`` is None.
    pathlib.Path
        The destination path (file written to disk), if ``dest`` is provided.

    Raises
    ------
    urllib.error.URLError / HTTPError on network or API failure.

    Example
    -------
    >>> raw = fetch_file(1234567)
    >>> text = raw.decode("utf-8", errors="replace")

    >>> path = fetch_file(1234567, dest="/tmp/dataset.tab")
    >>> import csv
    >>> with open(path) as fh:
    ...     rows = list(csv.DictReader(fh, delimiter='\\t'))
    """
    url = f"{server.rstrip('/')}/api/access/datafile/{file_id}"
    content = _get_bytes(url)
    if dest is None:
        return content
    dest = pathlib.Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(content)
    return dest


# ── Convenience: resolve file ID by name ─────────────────────────────────────

def find_file(
    pid: str,
    name_fragment: str,
    server: str = DATAVERSE_SERVER,
) -> Optional[dict]:
    """
    Return the first file metadata dict from ``list_files(pid)`` whose
    filename contains ``name_fragment`` (case-insensitive).

    Returns None if no match is found.

    Example
    -------
    >>> meta = find_file("doi:10.7910/DVN/HRK5HI", ".tab")
    >>> meta["dataFile"]["id"]
    """
    files = list_files(pid, server=server)
    frag = name_fragment.lower()
    for f in files:
        fname = f.get("dataFile", {}).get("filename", "")
        if frag in fname.lower():
            return f
    return None


# ── CLI ───────────────────────────────────────────────────────────────────────

def _cli_search(args):
    import sys
    hits = search(args.query, per_page=args.per_page, server=args.server)
    if not hits:
        print("No results.")
        return
    for h in hits:
        print(f"{h.get('global_id','?'):35s}  {h.get('name','')[:80]}")


def _cli_list(args):
    files = list_files(args.pid, server=args.server)
    if not files:
        print("No files found.")
        return
    for f in files:
        df = f.get("dataFile", {})
        print(
            f"{df.get('id','?'):>10}  {df.get('filename','?'):40s}"
            f"  {df.get('contentType','?'):30s}  {df.get('filesize',0):>12,} bytes"
        )


def _cli_fetch(args):
    import sys
    dest = pathlib.Path(args.dest) if args.dest else None
    result = fetch_file(args.file_id, dest=dest, server=args.server)
    if isinstance(result, pathlib.Path):
        print(f"Saved to: {result}")
    else:
        sys.stdout.buffer.write(result)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(
        description="Harvard Dataverse API helpers (search / list_files / fetch_file)."
    )
    parser.add_argument(
        "--server", default=DATAVERSE_SERVER,
        help="Dataverse server base URL (default: Harvard Dataverse)."
    )
    sub = parser.add_subparsers(dest="cmd")

    # search
    s = sub.add_parser("search", help="Search for datasets.")
    s.add_argument("query", help="Search query string.")
    s.add_argument("--per-page", type=int, default=20)

    # list
    lf = sub.add_parser("list", help="List files in a dataset by DOI.")
    lf.add_argument("pid", help="Persistent identifier (e.g. doi:10.7910/DVN/HRK5HI).")

    # fetch
    ff = sub.add_parser("fetch", help="Fetch a file by numeric file ID.")
    ff.add_argument("file_id", type=int, help="Dataverse numeric file ID.")
    ff.add_argument("--dest", default=None, help="Save to this path (else stdout).")

    args = parser.parse_args(argv)
    if args.cmd == "search":
        _cli_search(args)
    elif args.cmd == "list":
        _cli_list(args)
    elif args.cmd == "fetch":
        _cli_fetch(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
