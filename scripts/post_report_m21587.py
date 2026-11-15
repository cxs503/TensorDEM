import json, os, urllib.request
payload = json.load(open(os.environ.get('REPORT_JSON','/tmp/report.json')))
base = os.environ.get('TENSOR_API_BASE', 'http://127.0.0.1:8001/api/v1')
if not base.endswith('/api/v1'):
    base = base.rstrip('/') + '/api/v1'
login = json.dumps({"username": os.environ['TENSOR_AGENT_USER'],
                    "password": os.environ['TENSOR_AGENT_PASS']}).encode()
req = urllib.request.Request(base + '/auth/login', data=login,
                            headers={'Content-Type': 'application/json'}, method='POST')
body = json.loads(urllib.request.urlopen(req, timeout=180).read())
tok = body.get('access_token') or body.get('token')
assert tok, body
req = urllib.request.Request(base + '/tensor/reports', data=json.dumps(payload).encode(),
                            headers={'Content-Type': 'application/json',
                                     'Authorization': 'Bearer ' + tok}, method='POST')
print(urllib.request.urlopen(req, timeout=300).read().decode()[:600])
