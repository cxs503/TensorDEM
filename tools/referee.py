import json, psycopg

dsn = "dbname=tensorplatform user=postgres"
with psycopg.connect(dsn) as c, c.cursor() as cur:
    cur.execute("SELECT column_name FROM information_schema.columns WHERE table_name='tensor_benchmark_cases' ORDER BY ordinal_position")
    print("COLS:", [r[0] for r in cur.fetchall()])
    cur.execute("SELECT * FROM tensor_benchmark_cases WHERE anchor_slug=%s", ("CASE-DEM-MINDLIN-TANGENTIAL",))
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print("ROWS:", len(rows))
    for r in rows:
        print(json.dumps(dict(zip(cols, [str(x) for x in r])), ensure_ascii=False, indent=1))
    cur.execute("SELECT count(*) FROM tensor_benchmark_cases")
    print("TOTAL cases:", cur.fetchone()[0])
    cur.execute("SELECT anchor_slug FROM tensor_benchmark_cases ORDER BY anchor_slug")
    print([r[0] for r in cur.fetchall()])