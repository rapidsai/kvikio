# nVault public-repository credential test

This workflow exercises one public repository's GitHub OIDC identity against
nVault. It reads the Rapids Pulse SSA pair, exchanges it for an nSpect-scoped
OAuth token, and performs a read-only program-version request for
`NSPECT-1RDT-NZVP`. It performs no publication, scanner submission, rotation, or
nSpect data mutation.

Credential values remain in process memory. The workflow uses no stored
GitHub secret, human Vault token, shell credential file, artifact, or job
output. The client refuses redirects and suppresses response bodies on errors.

Both jobs target the NVIDIA-hosted ephemeral `linux-amd64-cpu4` runner used by
release workflows. The workflow runs only on `testing/nvault-ssa` and
`testing/nvault-ssa-denied`. The dedicated JWT role permits the positive branch
at one reviewed commit SHA, and binds the immutable repository and owner IDs,
workflow, ref, audience, and event. Its policy grants read-only access to the
Rapids Pulse SSA issue endpoint; administrative endpoints and other SSA
engines are checked for denial. The negative branch presents a real signed JWT
with a different ref and must be denied by nVault. The positive branch also
checks wrong-audience denial.

The GitHub-hosted runner job needs only `contents: read` and `id-token: write`.
No GitHub environment is used: the workload identity is restricted by the
exact branch, workflow, repository, owner, audience, event, and commit claims.

The JWT auth mount is `jwt/github-actions/literate-octo-apple`, with the role
name `rapids-kvikio-credential-test`. The role template is in `role.json`; set
its SHA to the reviewed commit before installing it. Do not commit a deployment
copy with a concrete SHA, because changing the file changes the commit being
trusted. Preserve the existing auth mount and policy; do not enroll or rotate
SSA clients for this test.

Local checks: `pytest -q ci/nvault/tests`, changed-file pre-commit checks, and
`actionlint`. Live evidence must include successful OIDC login, policy
isolation, credential retrieval, OAuth exchange, API read, and actual
wrong-audience and wrong-ref denials. Network failures do not count as
successful negative tests.
