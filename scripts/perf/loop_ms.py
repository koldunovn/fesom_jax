"""usage: loop_ms.py <run_core2_gpu log>...  Production-loop ms/step from run.progress lines: from the end of the last leading compile chunk to the loop end,
minus the device time of any later compile chunk (a new chunk length at a month end), over the steps in between."""
import re, sys
for f in sys.argv[1:]:
    t = open(f).read()
    rows = [tuple(float(x) for x in m) for m in re.findall(
        r"chunk (\d+)/\d+ step(\d+)->(\d+) host=([\d.]+)s device=([\d.]+)s \(loop elapsed ([\d.]+)s", t)]
    if not rows:
        print(f, "no chunk lines"); continue
    i0 = 0
    while i0 < len(rows) and rows[i0][4] >= 5:
        i0 += 1
    a = rows[i0 - 1]
    later = [r for r in rows[i0:] if r[4] >= 5]
    steps = rows[-1][2] - a[2] - sum(r[2] - r[1] for r in later)
    secs = rows[-1][5] - a[5] - sum(r[4] for r in later)
    ms = 1e3 * secs / steps
    print(f"{f.split('/')[-1]}: {ms:.2f} ms/step = {2700 / ms * 1e3 / 365:.0f} SYPD over {int(steps)} steps "
          f"(later compiles {[(int(r[0]), int(r[2]-r[1]), r[4]) for r in later]}); loop {rows[-1][5]:.0f} s")
