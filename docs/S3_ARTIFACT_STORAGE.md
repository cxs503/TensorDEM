# TensorDEM artifact upload to S3

The default solver path remains local and makes no cloud requests. After a completed
2-D or 3-D run has written `history.csv` and `trajectory.pt` or
`trajectory_3d.pt`, the run-manifest command can upload those exact files to S3
and verify the stored byte count and SHA-256 metadata.

## Install and configure

Install TensorPhysics from its repository in the same Python environment, including
its optional S3 dependency:

```bash
python -m pip install "boto3>=1.34"
python -m pip install "git+https://github.com/cxs503/TensorPhysics.git"
```

Configure credentials through boto3's normal credential provider chain. Do not put
access keys or secret keys in command arguments or manifests. For a compatible
endpoint, use `TENSORDEM_S3_ENDPOINT_URL` or `--s3-endpoint-url`. Set
`AWS_REGION` or `AWS_DEFAULT_REGION` as needed.

## Upload existing 2-D or 3-D results

```bash
# 2-D example; run the solver first
python -m tensordem --steps 100 --output results-2d
python -m tensordem.run_manifest --output results-2d --run-id ice-dem-2d-001 \
  --s3-artifact-store your-bucket --s3-prefix ice-dem

# 3-D example; run the solver first
python -m tensordem.dem3d_cli --nx 5 --ny 4 --nz 3 --steps 100 --output results-3d
python -m tensordem.run_manifest --output results-3d --run-id ice-dem-3d-001 \
  --s3-artifact-store your-bucket --s3-prefix ice-dem
```

The command uploads each solver output beneath
`s3://your-bucket/ice-dem/<run-id>/`, performs a remote HEAD verification after
each write, writes the local `run_manifest.json`, then uploads and verifies that
manifest too. The manifest's `outputs` array references the actual S3 object URIs,
sizes, and SHA-256 checksums. The local manifest itself is printed as
`manifest=...`; the remote manifest URI and digest are printed separately.

To register the manifest with TensorPlatform, append `--register-platform` and
configure `TENSORPLATFORM_URL`, `TENSORPLATFORM_API_TOKEN`, and
`TENSORPLATFORM_ORGANIZATION_ID`. This sends manifest metadata and S3 references
to the platform; it does not transfer the large solver output bytes to TensorPlatform.
The token must have `datasets:upload` permission in the target organization.

## Integrity and scientific status

- Existing identical object content is idempotent; conflicting content at the same
  object key is rejected by TensorPhysics' S3 store.
- Remote verification checks object existence, byte count and the SHA-256 value
  stored in object metadata. It does not re-download the complete object to
  independently recompute the digest.
- CI tests exercise the upload workflow with a fake S3 store. They do not prove
  connectivity or compatibility with a live AWS or third-party object store.
- The run manifest keeps `validation_status=not_run`. Solver completion and
  artifact integrity do not establish numerical benchmark success or physical
  validation of ice fracture or ship resistance.
- Use unique run IDs for new runs; object keys are treated as immutable.
