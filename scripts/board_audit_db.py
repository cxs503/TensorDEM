#!/usr/bin/env python3
"""Board-audit: registered referee rows + mission goal for m16147.

DSN is read from the platform .env; it is never printed.
Output: JSON (zero hand-copy into the verdict).
"""
import json
import re

ENV = "/root/tensor-ecosystem/TensorPlatform/.env"


def dsn():
    for line in open(ENV):
        if line.startswith("DATABASE_URL="):
            return line.split("=", 1)[1].strip().replace("postgresql+psycopg://",
                                                         "postgresql://")
    raise SystemExit("no DATABASE_URL")


def main():
    import psycopg
    out = {}
    with psycopg.connect(dsn()) as c, c.cursor() as cur:
        cur.execute("SELECT id, anchor_slug, case_name, metric, tolerance, repo, status, "
                    "created_at FROM tensor_benchmark_cases "
                    "WHERE anchor_slug=%s ORDER BY id", ("CASE-DEM-MINDLIN-TANGENTIAL",))
        cols = [d.name for d in cur.description]
        out["registered_rows"] = [dict(zip(cols, r)) for r in cur.fetchall()]
        for pat in ("%MINDLIN%", "%DEM%"):
            cur.execute("SELECT count(*) FROM tensor_benchmark_cases WHERE anchor_slug LIKE %s",
                        (pat,))
            out["count_like_" + pat.strip("%").lower()] = cur.fetchone()[0]
        cur.execute("SELECT id, repo, goal, status FROM tensor_dev_missions WHERE id=16147")
        r = cur.fetchone()
        out["mission"] = ({"id": r[0], "repo": r[1], "status": r[3],
                           "goal_excerpt": (r[2] or "")[:2000]} if r else None)
        # prior verdicts on this anchor (any mission)
        cur.execute("SELECT count(*) FROM information_schema.tables "
                    "WHERE table_name='tensor_maturity_verdicts'")
        out["verdict_table_exists"] = cur.fetchone()[0] == 1
        if out["verdict_table_exists"]:
            cur.execute("SELECT repo, scope, mission_id, level, verdict_by, created_at "
                        "FROM tensor_maturity_verdicts WHERE scope LIKE %s ORDER BY id",
                        ("%MINDLIN-TANGENTIAL%",))
            out["prior_verdicts"] = cur.fetchall()
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()