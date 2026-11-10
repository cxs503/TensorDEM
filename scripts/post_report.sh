#!/bin/bash
# Post report to Tensor API
TOKEN=$(curl -s -X POST "$TENSOR_API_BASE/auth/login" \
  -H "Content-Type: application/json" \
  -d "{\"username\":\"$TENSOR_AGENT_USER\",\"password\":\"$TENSOR_AGENT_PASS\"}" \
  | python3 -c "import sys,json; print(json.load(sys.stdin).get('access_token',''))")

if [ -z "$TOKEN" ]; then
  echo "Failed to get token"
  exit 1
fi

echo "Got token, posting report..."
curl -s -X POST "$TENSOR_API_BASE/tensor/reports" \
  -H "Authorization: Bearer $TOKEN" \
  -H "Content-Type: application/json" \
  -d @/tmp/report.json

echo ""
echo "Report posted."
