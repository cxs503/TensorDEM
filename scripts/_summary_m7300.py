"""Summarise campaign case JSONs into a table (m7300)."""
import json, glob, os, sys

rows = []
for f in sorted(glob.glob('runs/p70n03/*.json')):
    d = json.load(open(f))
    n = os.path.basename(f)[:-5]
    if d.get('status') == 'error':
        rows.append((n, None))
        continue
    rows.append((n, d))

ok = [r for r in rows if r[1]]
print(len(ok), "ok /", len(rows), "cases")
print("%-24s %5s %5s %5s %5s %-7s %-5s %9s %11s %7s" % (
    "case", "s", "phi", "Gam", "f", "geom", "drive", "S", "v_up", "ovl"))
for n, d in ok:
    c, s = d['case'], d['summary']
    def fmt(v, f="%.4f"):
        return f % v if isinstance(v, (int, float)) else str(v)
    print("%-24s %5.1f %5.1f %5.1f %5.0f %-7s %-5s %9s %11s %7s" % (
        n, c['size_ratio'], c['large_volume_fraction'], c['gamma'], c['frequency'],
        c['geometry'], c['drive'], fmt(s.get('S')), fmt(s.get('v_up'), '%.3e'),
        fmt(d['packing'].get('kn_overlap_ratio'), '%.3f')))
errs = [n for n, d in rows if not d]
print("failed:", len(errs))
print(" ".join(errs))