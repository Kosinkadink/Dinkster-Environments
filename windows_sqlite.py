"""Build the pinned Windows SQLite WAL fix and verify mapped-shm recovery."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import mmap
import shutil
import sqlite3
import subprocess
import sys
import tarfile
import tempfile
import tomllib
import urllib.request
import zipfile
from pathlib import Path

SOURCE = {
    "url": "https://www.sqlite.org/2026/sqlite-amalgamation-3530100.zip",
    "size": 2938639,
    "sha256": "36ad6e7f38540a3b21a2ac36340833f0a9e426bc1c752751c3ba669466827eae",
}
FIX = "https://sqlite.org/src/info/7c103a0b7d509dce58aa87518e13213451e5f70938ff08aa55accd645e3b2ff3"
FORUM = "https://sqlite.org/forum/forumpost/f9c68d01e2d7f638"
FLAGS = [
    "-target",
    "x86_64-windows-gnu",
    "-shared",
    "-O2",
    "-s",
    "-DSQLITE_API=__declspec(dllexport)",
    "-DSQLITE_DEFAULT_AUTOVACUUM=1",
    "-DSQLITE_DEFAULT_RECURSIVE_TRIGGERS=1",
    "-DSQLITE_ENABLE_DBSTAT_VTAB",
    "-DSQLITE_ENABLE_GEOPOLY",
    "-DSQLITE_ENABLE_FTS3",
    "-DSQLITE_ENABLE_FTS3_PARENTHESIS",
    "-DSQLITE_ENABLE_FTS4",
    "-DSQLITE_ENABLE_FTS5",
    "-DSQLITE_ENABLE_MATH_FUNCTIONS",
    "-DSQLITE_ENABLE_RTREE",
    "-DSQLITE_OMIT_AUTOINIT",
    "-DSQLITE_THREADSAFE=1",
]

# Exact os_win.c changes from the upstream check-in, not a truncate-error retry.
ZERO_SHARED_MEMORY = """/*
** Handle h is open on the *-shm file. This function zeroes the first
** 8 bytes of the file. If SQLITE_ENABLE_SETLK_TIMEOUT is defined, h may
** have been opened with FILE_FLAG_OVERLAPPED.
*/
static int winZeroSharedMemory(HANDLE h){
  u8 aZero[8] = {0,0,0,0,0,0,0,0};
  DWORD nWrite = 0;
  OVERLAPPED ovlp;
  memset(&ovlp, 0, sizeof(OVERLAPPED));
  if( !osWriteFile(h, aZero, sizeof(aZero), &nWrite, &ovlp) ){
    /* If the file was opened with FILE_FLAG_OVERLAPPED, the write may still
    ** be pending. Wait for it with GetOverlappedResult().  */
    if( osGetLastError()!=ERROR_IO_PENDING
     || !GetOverlappedResult(h, &ovlp, &nWrite, TRUE)
    ){
      nWrite = 0;
    }
  }
  return (nWrite==sizeof(aZero) ? SQLITE_OK : SQLITE_IOERR_WRITE);
}

"""
TRUNCATE = """      rc = winHandleTruncate(h, 0);"""
RECOVER = """      if( SQLITE_OK!=winHandleTruncate(h, 0) ){
        /* Sometimes the truncate operation may fail because some other
        ** process is still holding an open mapping. So try to write
        ** zeroes into the start of the file instead. */
        rc = winZeroSharedMemory(h);
      }"""


def probe() -> None:
    """Require WAL reads and integrity while an old shm mapping remains open."""
    if sys.platform != "win32":
        raise RuntimeError("The mapped-shm gate requires native Windows")
    print(f"CPython {sys.version}; SQLite {sqlite3.sqlite_version}", flush=True)
    try:
        with tempfile.TemporaryDirectory(prefix="dinkster-sqlite-") as temporary:
            path = Path(temporary) / "wal.sqlite"
            db = sqlite3.connect(path)
            try:
                if db.execute("PRAGMA journal_mode=WAL").fetchone() != ("wal",):
                    raise RuntimeError("SQLite did not enable WAL")
                db.execute("CREATE TABLE items (value TEXT)")
                db.execute("INSERT INTO items VALUES ('persisted')")
                db.commit()
                with Path(str(path) + "-shm").open("r+b") as shm:
                    shm.truncate(65536)
                    with mmap.mmap(shm.fileno(), 0) as mapping:
                        if len(mapping) != 65536:
                            raise RuntimeError(
                                "SQLite probe requires a 64 KiB shm mapping"
                            )
                        db.close()
                        for state in ("mapped", "released"):
                            if state == "released":
                                mapping.close()
                            connection = sqlite3.connect(path)
                            try:
                                rows = connection.execute(
                                    "SELECT * FROM items"
                                ).fetchall()
                                integrity = connection.execute(
                                    "PRAGMA integrity_check"
                                ).fetchall()
                                if rows != [("persisted",)] or integrity != [("ok",)]:
                                    raise RuntimeError(f"{state}: {rows}, {integrity}")
                                print(f"{state}: persisted; integrity_check=ok")
                            finally:
                                connection.close()
            finally:
                db.close()
    except Exception as error:
        raise RuntimeError(
            f"Windows SQLite mapped-shm recovery failed: {error}; {FORUM}; fix {FIX}"
        ) from error


def build(output: Path) -> dict:
    """Compile the official, hash-verified amalgamation with the upstream fix."""
    if output.exists():
        raise ValueError("SQLite output must be a new directory")
    compiler = [sys.executable, "-m", "ziglang"]
    version = subprocess.check_output([*compiler, "version"], text=True).strip()
    if version != "0.14.1":
        raise ValueError("SQLite build requires Zig 0.14.1")
    with urllib.request.urlopen(SOURCE["url"], timeout=60) as response:
        archive = response.read()
    if (
        len(archive) != SOURCE["size"]
        or hashlib.sha256(archive).hexdigest() != SOURCE["sha256"]
    ):
        raise ValueError("SQLite amalgamation size/hash mismatch")
    with tempfile.TemporaryDirectory(prefix="dinkster-sqlite-build-") as temporary:
        work = Path(temporary)
        with zipfile.ZipFile(io.BytesIO(archive)) as zipped:
            source = zipped.read("sqlite-amalgamation-3530100/sqlite3.c").decode()
        anchor = (
            "/*\n** The DMS lock has not yet been taken on the shm file associated with"
        )
        if source.count(anchor) != 1 or source.count(TRUNCATE) != 1:
            raise ValueError("SQLite upstream patch context does not match")
        source = source.replace(anchor, ZERO_SHARED_MEMORY + anchor)
        source = source.replace(TRUNCATE, RECOVER)
        (work / "sqlite3.c").write_bytes(source.encode("ascii"))
        subprocess.run(
            [*compiler, "cc", *FLAGS, "sqlite3.c", "-o", "sqlite3.dll"],
            cwd=work,
            check=True,
        )
        output.mkdir(parents=True)
        shutil.copyfile(work / "sqlite3.dll", output / "sqlite3.dll")
    dll = output / "sqlite3.dll"
    metadata = {
        "version": "3.53.1",
        "source": SOURCE,
        "fix": FIX,
        "compiler": f"zig {version}",
        "flags": FLAGS,
        "patchedSourceSha256": hashlib.sha256(source.encode("ascii")).hexdigest(),
        "filename": "sqlite3.dll",
        "path": "python/DLLs/sqlite3.dll",
        "size": dll.stat().st_size,
        "sha256": hashlib.sha256(dll.read_bytes()).hexdigest(),
    }
    (output / "sqlite.json").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="ascii"
    )
    return metadata


def install(directory: Path, python: Path) -> None:
    """Replace only SQLite in a Windows distribution, then run its functional gate."""
    metadata = json.loads((directory / "sqlite.json").read_text())
    dll = directory / "sqlite3.dll"
    if (
        dll.stat().st_size != metadata["size"]
        or hashlib.sha256(dll.read_bytes()).hexdigest() != metadata["sha256"]
    ):
        raise ValueError("SQLite override size/hash mismatch")
    # Resolve virtual environments to their base interpreter before replacing the DLL.
    prefix = Path(
        subprocess.check_output(
            [str(python), "-I", "-c", "import sys; print(sys.base_prefix)"], text=True
        ).strip()
    )
    destination = prefix / "DLLs/sqlite3.dll"
    if not destination.is_file():
        raise ValueError("Windows interpreter lacks a separate DLLs/sqlite3.dll")
    version = subprocess.check_output(
        [str(python), "-I", "-c", "import sqlite3; print(sqlite3.sqlite_version)"],
        text=True,
    ).strip()
    if version != metadata["version"]:
        raise ValueError("SQLite override must match the interpreter's SQLite version")
    shutil.copyfile(dll, destination)
    subprocess.run(
        [str(python), "-I", str(Path(__file__).resolve()), "probe"],
        check=True,
        timeout=60,
    )


def acquire(config_path: Path, output: Path) -> Path:
    """Acquire a private, patched Windows interpreter without changing uv's cache."""
    if sys.platform != "win32":
        raise ValueError("Windows runtime acquisition requires native Windows")
    config = tomllib.loads(config_path.read_text(encoding="ascii"))
    interpreter = config["interpreters"]["windows"]
    expected = interpreter["sqlite"]
    python = output / "python/python.exe"
    if output.exists():
        dll = output / "python/DLLs/sqlite3.dll"
        if (
            dll.stat().st_size != expected["size"]
            or hashlib.sha256(dll.read_bytes()).hexdigest() != expected["sha256"]
        ):
            raise ValueError("Existing Windows runtime differs from pinned SQLite")
        subprocess.run(
            [str(python), "-I", str(Path(__file__).resolve()), "probe"],
            check=True,
            timeout=60,
        )
        return python
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix=".windows-runtime-", dir=output.parent
    ) as temporary:
        work = Path(temporary)
        url = (
            "https://github.com/astral-sh/python-build-standalone/releases/download/"
            f"{config['base']['release']}/{interpreter['filename']}"
        )
        with urllib.request.urlopen(url, timeout=60) as response:
            archive = response.read()
        if (
            len(archive) != interpreter["size"]
            or hashlib.sha256(archive).hexdigest() != interpreter["sha256"]
        ):
            raise ValueError("Windows interpreter size/hash mismatch")
        payload = work / "payload"
        payload.mkdir()
        with tarfile.open(fileobj=io.BytesIO(archive), mode="r:gz") as tar:
            tar.extractall(payload, filter="data")
        override = work / "sqlite"
        if build(override) != expected:
            raise ValueError("Built SQLite differs from pinned override")
        install(override, payload / "python/python.exe")
        payload.rename(output)
    return python


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("probe")
    compile_command = commands.add_parser("build")
    compile_command.add_argument("--output", type=Path, required=True)
    patch_command = commands.add_parser("install")
    patch_command.add_argument("--directory", type=Path, required=True)
    patch_command.add_argument("--python", type=Path, required=True)
    acquire_command = commands.add_parser("acquire")
    acquire_command.add_argument("--config", type=Path, required=True)
    acquire_command.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "probe":
        probe()
    elif args.command == "build":
        print(json.dumps(build(args.output.resolve()), sort_keys=True))
    elif args.command == "acquire":
        print(acquire(args.config.resolve(), args.output.resolve()))
    else:
        install(args.directory.resolve(), args.python.resolve())


if __name__ == "__main__":
    main()
