"""Patient-ID extraction and cross-sequence alignment utilities."""
import os
import re
from collections import defaultdict

PID_REGEX = re.compile(r"(PCa-[^_]+)")


def extract_pid(path: str) -> str:
    """Extract a ``PCa-XXXXX`` patient id from any of the sequence filenames."""
    base = os.path.basename(path)
    match = PID_REGEX.search(base)
    if match is None:
        raise ValueError(f"Could not extract patient_id from filename: {base}")
    return match.group(1)


def _index_by_pid(paths, sequence: str, strict: bool = True) -> dict:
    """Return ``{pid: path}`` for one sequence. Raises on duplicates when strict."""
    bucket = defaultdict(list)
    for p in paths:
        bucket[extract_pid(p)].append(p)

    dups = {pid: lst for pid, lst in bucket.items() if len(lst) > 1}
    if dups and strict:
        sample = ", ".join(f"{pid}({len(lst)})" for pid, lst in list(dups.items())[:5])
        raise ValueError(f"Duplicate {sequence} files for: {sample}")

    return {pid: sorted(lst)[0] for pid, lst in bucket.items()}


def align_sequences(t2w_list, dwi_list, adc_list, clinical_list, strict: bool = True):
    """Align per-sequence file lists by patient id.

    Returns the four lists in the same order plus the list of aligned patient ids.
    When ``strict=True`` raises if any patient is missing any sequence.
    """
    idx = {
        "t2w": _index_by_pid(t2w_list, "t2w", strict),
        "dwi": _index_by_pid(dwi_list, "dwi", strict),
        "adc": _index_by_pid(adc_list, "adc", strict),
        "clinical": _index_by_pid(clinical_list, "clinical", strict),
    }
    pids_all = set().union(*idx.values())
    pids_ok = set.intersection(*(set(v) for v in idx.values()))

    if strict and pids_ok != pids_all:
        missing = {k: sorted(pids_all - set(v))[:10] for k, v in idx.items() if pids_all - set(v)}
        raise ValueError(f"Missing sequences (first 10 per key): {missing}")

    aligned = sorted(pids_ok)
    return (
        [idx["t2w"][p] for p in aligned],
        [idx["dwi"][p] for p in aligned],
        [idx["adc"][p] for p in aligned],
        [idx["clinical"][p] for p in aligned],
        aligned,
    )
