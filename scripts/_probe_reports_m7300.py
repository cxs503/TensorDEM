import json, os, urllib.request

BASE = os.environ["TENSOR_API_BASE"]
tok = open("/tmp/tok.txt").read().strip()


def get(path):
    req = urllib.request.Request(BASE + path, headers={"Authorization": "Bearer " + tok})
    try:
        with urllib.request.urlopen(req) as r:
            return r.status, json.load(r)
    except Exception as e:  # noqa
        return "ERR", str(e)


for p in ["/missions/7300", "/tensor/missions/7300", "/missions?ids=7300"]:
    s, d = get(p)
    print(p, s, json.dumps(d, ensure_ascii=False)[:2500])
    print()