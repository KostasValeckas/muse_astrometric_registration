"""
Logging module.

Taken from: https://github.com/KostasValeckas/PyLongslit/blob/main/pylongslit/logger.py
"""

import logging
from colorama import Fore, Style, init
from datetime import datetime
from pathlib import Path


def get_logger(output_path, logger_name="muse_astrometric_registration"):
    """
    Create and configure a logger.

    Parameters
    ----------
    output_path : str or Path
        The directory where the log file will be saved.

    logger_name : str, optional
        The name of the logger. Default is "muse_astrometric_registration".

    Returns
    -------
    logger : logging.Logger
        Configured logger object.
    """

    # Initialize colorama
    init(autoreset=True)

    # Create a custom logger
    logger = logging.getLogger(logger_name)

    # Close and remove leftover handlers. Iterate over a copy: removing from
    # the list being iterated skips every second handler, which then keeps
    # writing the next run into the previous log file
    for handler in list(logger.handlers):
        handler.close()
        logger.removeHandler(handler)

    # Configure logging level
    logger.setLevel(logging.INFO)

    output_path = Path(output_path) if isinstance(output_path, str) else output_path
    log_file_path = output_path / datetime.now().strftime(
        "aligment_log_%Y-%m-%d_%H-%M-%S.log"
    )

    # we add different information when we print to console and when we write to
    # file, so we need two handlers
    fh = logging.FileHandler(log_file_path)
    ch = logging.StreamHandler()

    # Create a custom formatter - this allows different colors for different log levels
    class CustomFormatter(logging.Formatter):
        def format(self, record):
            # color a copy of the record - the record is shared between the
            # handlers, and changing it here would put the color codes into
            # the log file as well
            record = logging.makeLogRecord(record.__dict__)
            if record.levelno == logging.INFO:
                record.levelname = f"{Fore.GREEN}{record.levelname}{Style.RESET_ALL}"
            elif record.levelno == logging.WARNING:
                record.levelname = f"{Fore.YELLOW}{record.levelname}{Style.RESET_ALL}"
            elif record.levelno == logging.ERROR:
                record.levelname = f"{Fore.RED}{record.levelname}{Style.RESET_ALL}"
            elif record.levelno == logging.CRITICAL:
                record.levelname = (
                    f"{Fore.RED}{record.levelname}{Style.BRIGHT}{Style.RESET_ALL}"
                )
            return super().format(record)

    # For filelogging we add the dates, for console logging we don't
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    color_formatter = CustomFormatter(
        "%(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
    )

    fh.setFormatter(formatter)
    ch.setFormatter(color_formatter)

    # Add both handlers to the logger
    logger.addHandler(fh)
    logger.addHandler(ch)

    # Log messages
    logger.info("Logger initialized. Log will be saved in " + fh.baseFilename + ".")

    return logger


def format_elapsed(seconds):
    """
    Format a duration in seconds as "1 h 02 min 03.4 s" (leading zero units
    are left out), for the execution time log lines.
    """
    hours, rest = divmod(float(seconds), 3600)
    minutes, secs = divmod(rest, 60)

    if hours >= 1:
        return f"{int(hours)} h {int(minutes):02d} min {secs:04.1f} s"
    if minutes >= 1:
        return f"{int(minutes)} min {secs:04.1f} s"
    return f"{secs:.1f} s"


def log_run_environment(logger):
    """
    Log what is needed to reproduce / debug a run: the git commit of the code
    (and whether it has local changes), host, Python and package versions, the
    command line, the working directory and the batch job id (SLURM).

    Never raises: anything that cannot be determined is logged as unknown.
    """
    import os
    import platform
    import subprocess
    import sys

    repo_dir = Path(__file__).resolve().parent

    def _git(*args):
        try:
            out = subprocess.run(
                ["git", "-C", str(repo_dir), *args],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return out.stdout.strip() if out.returncode == 0 else None
        except Exception:
            return None

    commit = _git("log", "-1", "--format=%h %cd %s", "--date=iso")
    branch = _git("rev-parse", "--abbrev-ref", "HEAD")
    changes = _git("status", "--porcelain", "--untracked-files=no")

    if commit is None:
        logger.warning(f"Code version: unknown (no git repository at {repo_dir}).")
    else:
        logger.info(f"Code version: {commit} (branch {branch}), from {repo_dir}")
        if changes:
            logger.warning(
                "Code has uncommitted local changes:\n      "
                + "\n      ".join(changes.splitlines())
            )

    logger.info(
        f"Host: {platform.node()}, Python {platform.python_version()} "
        f"({sys.executable})"
    )

    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        version = None

    packages = (
        "numpy",
        "scipy",
        "astropy",
        "reproject",
        "scikit-image",
        "matplotlib",
        "astroalign",
        "sep",
        "spacepylot",
    )
    versions = []
    for package in packages:
        try:
            versions.append(f"{package} {version(package)}")
        except Exception:
            versions.append(f"{package} ?")
    logger.info("Packages: " + ", ".join(versions))

    logger.info(f"Command line: {' '.join(sys.argv)}")
    logger.info(f"Working directory: {os.getcwd()}")

    job = {
        key: os.environ[key]
        for key in (
            "SLURM_JOB_ID",
            "SLURM_JOB_NAME",
            "SLURM_CPUS_PER_TASK",
            "SLURM_MEM_PER_NODE",
            "OMP_NUM_THREADS",
        )
        if key in os.environ
    }
    if job:
        logger.info("Batch environment: " + ", ".join(f"{k}={v}" for k, v in job.items()))


def exit_on_sigterm():
    """
    Turn SIGTERM (sent e.g. by SLURM when the time limit is reached, before
    the job is killed) into a SystemExit, so the pipeline logs the run as
    ABORTED with the traceback (where it was) instead of dying silently.
    """
    import signal

    def _handler(signum, frame):
        raise SystemExit(f"Received signal {signum} (SIGTERM), stopping.")

    try:
        signal.signal(signal.SIGTERM, _handler)
    except (ValueError, OSError):
        # not in the main thread / not supported: keep the default
        pass


def peak_memory_mb():
    """
    Peak resident memory of this process in MB (Linux / macOS), or NaN if it
    cannot be determined. Logged per pointing, so a memory leak (or a job
    killed for memory) can be traced in the log.
    """
    try:
        import resource
        import sys

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        # kilobytes on Linux, bytes on macOS
        return peak / (1024.0**2) if sys.platform == "darwin" else peak / 1024.0
    except Exception:
        return float("nan")
