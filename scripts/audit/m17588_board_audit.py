#!/usr/bin/env python3
"""Board-audit evidence collector for m17500 (anchor XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE).

Reads the platform DB (DSN from TensorPlatform .env, never printed) and emits a
single JSON blob: registered benchmark referee rows, mission row, mission
result/report pointers. Zero hand-copy into the verdict.
"""
import json
import sys

ENV = "/root/tensor-ecosystem/TensorPlatform/.env"
ANCHOR = "XVAL-DEM-MU-I-CONTACT-PROXY-NEGATIVE"


def dsn():
    for line in open(ENV):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().replace(
                "postgresql+psycopg://", "postgresql://")
    raise SystemExit("no DATABASE_URL")


def main():
    import psycopg
    out = {"anchor": ANCHOR}
    with psycopg.connect(dsn()) as c, c.cursor() as cur:
        # 1) registered referee rows
        cur.execute(
            "SELECT id, anchor_slug, case_name, metric, tolerance, repo, status, created_at "
            "FROM tensor_benchmark_cases WHERE anchor_slug=%s ORDER BY id", (ANCHOR,))
        cols = [d.name for d in cur.description]
        out["benchmark_cases"] = [dict(zip(cols, r)) for r in cur.fetchall()]

        # 2) mission row
        cur.execute("SELECT id, repo, goal, status FROM tensor_dev_missions WHERE id=17500")
        r = cur.fetchone()
        out["mission_17500"] = ({"id": r[0], "repo": r[1], "status": r[3],
                                 "goal": r[2]} if r else None)

        # 3) reports
        cur.execute("SELECT id, mission_id, title, scenario_key, created_at, "
                    "length(markdown) FROM tensor_reports WHERE mission_id=17500 ORDER BY id")
        cols = [d.name for d in cur.description]
        out["reports"] = [dict(zip(cols, r)) for r in cur.fetchall()]

        # 4) verdicts already on this anchor / mission
        cur.execute("SELECT id, repo, scope, mission_id, level, verdict_by, created_at "
                    "FROM tensor_maturity_verdicts WHERE mission_id=17500 OR scope LIKE %s "
                    "ORDER BY id", ("%MU-I-CONTACT-PROXY%",))
        cols = [d.name for d in cur.description]
        out["prior_verdicts"] = [dict(zip(cols, r)) for r in cur.fetchall()]

        # 5) runs / evidence registries if present
        for tbl, q in (
            ("tensor_run_artifacts",
             "SELECT * FROM tensor_run_artifacts WHERE mission_id=17500 LIMIT 20"),
            ("tensor_mission_evidence",
             "SELECT * FROM tensor_mission_evidence WHERE mission_id=17500 LIMIT 20"),
        ):
            cur.execute("SELECT to_regclass(%s)", (tbl,))
            if cur.fetchone()[0]:
                try:
                    cur.execute(q)
                    cols = [d.name for d in cur.description]
                    out[tbl] = [dict(zip(cols, r)) for r in cur.fetchall()]
                except Exception as e:  # noqa: BLE001
                    out[tbl] = {"error": str(e)[:200]}

        # 6) mission log / events
        for tbl in ("tensor_mission_logs", "tensor_events", "tensor_mission_events"):
            cur.execute("SELECT to_regclass(%s)", (tbl,))
            if cur.fetchone()[0]:
                try:
                    cur.execute(
                        "SELECT * FROM " + tbl + " WHERE mission_id=17500 "
                        "ORDER BY id DESC LIMIT 15")
                    cols = [d.name for d in cur.description]
                    out[tbl] = [dict(zip(cols, r)) for r in cur.fetchall()]
                except Exception as e:  # noqa: BLE001
                    out[tbl] = {"error": str(e)[:200]}
    json.dump(out, sys.stdout, indent=1, default=str)
    print()


if __name__ == "__main__":
    main()