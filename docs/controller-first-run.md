# First-run controller setup (creating the admin account)

A freshly deployed controller has **no users** and refuses to do anything
until one admin exists. Until then the dashboard serves the setup form and
`GET /api/system/setup-state` returns `{"needs_setup": true}`.

## The bootstrap token is ONE token printed across two lines

On first start the controller prints a one-time bootstrap token in a box:

```
  ┌─────────────────────────────────────────────────────────┐
  │  EchoMuse first-run setup                               │
  │                                                         │
  │  No users found. Visit /setup to create an admin        │
  │  account using this one-time token:                     │
  │                                                         │
  │  <first 32 hex characters>  │                     │
  │  <second 32 hex characters>  │                     │
  │                                                         │
  │  This token is not stored and will not appear again.    │
  └─────────────────────────────────────────────────────────┘
```

Those are **two halves of a single 64-character token**, printed split at
position 32 for width — the source does `token[:32]` and `token[32:]`.
Copying only the first line gives `401 {"error":"Invalid setup token"}`.
**Join the two lines with no whitespace** to get the token.

The token is not stored, so it cannot be recovered after the fact — it is
regenerated whenever the controller starts with no users, which means a
container restart mints a fresh one.

## Creating the account over HTTP

`POST /api/setup` with `{token, username, password}`, returns `201` plus
`{token, role}` so the caller is logged in immediately:

```bash
curl -sX POST http://<controller>:8768/api/setup \
  -H 'Content-Type: application/json' \
  -d '{"token":"<64-char token>","username":"<name>","password":"<password>"}'
```

Rules the controller enforces: the token must match exactly, setup can
only run once, and the password must be at least 8 characters.

**Do not put the owner's password in a chat message.** Generate it,
create the account, and write the credentials to a file with owner-only
permissions (mode `0600`) that the owner can read directly. Report where
the file is, not the secret.

## Reading the controller's health afterwards

| Signal | Meaning |
|---|---|
| `GET /api/system/setup-state` → `{"needs_setup": false}` | admin exists |
| `GET :8768/` → HTTP 200 | dashboard served |
| `GET :8767/` | **HTTP 426 Upgrade Required** | correct — it is a WebSocket endpoint and rejects a plain HTTP request |
| log: `mDNS advertising <name>._emcontroller._tcp.local → <ip>:8767 (tls_port=8770)` | discovery is live |
| log: `EchoMuse Controller ready — waiting for devices` | fully started |

The dashboard's own release poller logs `Latest controller release: vX
(running vY, current)` — if those disagree, the image is older than the
upstream release, which is fine but worth knowing.

## The version env var overrides the image's own version

`.env.example` ships `ESPHOME_PROJECT_VERSION=2.0.0` as a placeholder.
The env var **wins** over the version baked into the image
(`em_esphome.py`: `os.environ.get("ESPHOME_PROJECT_VERSION",
_CONTROLLER_VERSION)`), and Home Assistant displays that value as the
satellite's version. Leave the placeholder in place and HA shows the
device as version 2.0.0 forever and updates misreport — set it to the
controller release tag the image reports (check
`docker image inspect ... .Config.Env` for `EM_CONTROLLER_VERSION`).
