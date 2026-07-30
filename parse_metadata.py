# -*- coding: utf-8 -*-
"""
parse_metadata.py

Parses DICOM filenames into study metadata:
  - family:    'WT' (WT/WF/WM strain codes) or 'OX' (OX/OF/OM strain codes)
  - treatment: 'VEH', 'PF', 'ISO', or 'PF_ISO' (combined PF+ISO)
  - day:       0, 7, or 14
  - animal_id: unique per physical animal = treatment + strain_code
               (e.g. "ISO_OM10"), tracked across day0/7/14 scans
  - strain_code: OF/OM/OX/WF/WM/WT

Import get_metadata() / parse_all() from this module; not meant to be run
standalone in isolation, though `python parse_metadata.py <dir>` will print
a parse-coverage report for a folder of DICOM files.
"""

import os
import re
import sys

STRAIN_PATTERN = re.compile(r'(O[FMX]|W[FMT])\s?(\d+)')
DAY_PATTERN = re.compile(r'DAY\s*(\d+)', re.IGNORECASE)


def get_metadata(filename):
    """Returns a dict with family/treatment/day/strain_code/animal_number/animal_id,
    or None if the filename couldn't be parsed (missing strain code or day)."""
    upper = filename.upper()

    strain_m = STRAIN_PATTERN.search(upper)
    day_m = DAY_PATTERN.search(upper)
    if not strain_m or not day_m:
        return None

    strain_code = strain_m.group(1)
    animal_number = strain_m.group(2)
    day = int(day_m.group(1))
    family = 'WT' if strain_code[0] == 'W' else 'OX'

    # Treatment: check compound "PF ISO" / "PF_ISO" / "PFISO" before standalone PF/ISO
    prefix = upper[:strain_m.start()]
    if re.search(r'PF\s*ISO', prefix):
        treatment = 'PF_ISO'
    elif 'VEH' in prefix:
        treatment = 'VEH'
    elif 'ISO' in prefix:
        treatment = 'ISO'
    elif 'PF' in prefix:
        treatment = 'PF'
    else:
        return None

    animal_id = f"{treatment}_{strain_code}{animal_number}"

    return {
        'family': family,
        'treatment': treatment,
        'day': day,
        'strain_code': strain_code,
        'animal_number': animal_number,
        'animal_id': animal_id,
    }


def parse_all(directory):
    """Returns (parsed: list of (filename, metadata_dict), unparsed: list of filenames)"""
    files = [f for f in os.listdir(directory) if f.endswith('.dcm')]
    parsed = []
    unparsed = []
    for f in files:
        meta = get_metadata(f)
        if meta is None:
            unparsed.append(f)
        else:
            parsed.append((f, meta))
    return parsed, unparsed


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python parse_metadata.py <dicom_folder>")
        sys.exit(1)

    from collections import Counter
    directory = sys.argv[1]
    parsed, unparsed = parse_all(directory)

    print(f"Total files: {len(parsed) + len(unparsed)}")
    print(f"Parsed: {len(parsed)}")
    print(f"Unparsed: {len(unparsed)}")
    if unparsed:
        print("\nUnparsed files:")
        for f in unparsed:
            print(f"  {f}")

    print("\nFamily x Treatment x Day counts:")
    counts = Counter((m['family'], m['treatment'], m['day']) for _, m in parsed)
    for key in sorted(counts):
        print(f"  {key}: {counts[key]}")

    n_animals = len(set(m['animal_id'] for _, m in parsed))
    print(f"\nUnique animal_ids: {n_animals}")
