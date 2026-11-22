# Hosting the EchoMuse controller

The controller is the service the Echo talks to. It serves a dashboard on
`API_PORT` (8768), a device link on `SERVER_PORT` (8767), an encrypted
device link on `SERVER_TLS_PORT` (8770), advertises itself over **mDNS**,
and opens one ESPHome TCP port per device (16001+).

## The constraint that decides where it runs

The provisioning wizard is served from the controller's dashboard and
talks to the Echo over **WebUSB**, and browsers only permit WebUSB on a
**secure origin** — `http://localhost` or HTTPS. The controller's TLS
listener is for the *device* link; **the dashboard port is plain HTTP with
no HTTPS option**. So the browser must sit on the same machine as the
controller and reach it at `http://localhost:<port>`, unless the origin is
made secure another way.

The dashboard's own pre-flight panel names the origin and the fix
(`controller/static/dashboard.jsx`, `webUsbBlocked()`):

- non-ingress: *"Use http://localhost:8768 on this computer, or allow
  `${origin}` at chrome://flags/#unsafely-treat-insecure-origin-as-secure
  and relaunch the browser."*
- under the add-on: *"Serve Home Assistant over HTTPS, or allow
  `${origin}` at chrome://flags/#unsafely-treat-insecure-origin-as-secure
  and relaunch the browser. The controller's own address does not cover
  this one."*

**Chrome's allowlist matches scheme, host AND port exactly.** One entry
for some other address does not cover it, and a wrong entry gives no clue
why (#169). The flag is stored in Chrome's `Local State` under
`browser.enabled_labs_experiments` as
`unsafely-treat-insecure-origin-as-secure=<origin>`.

## Docker on a Raspberry Pi (ARM64)

The published image `ghcr.io/wilbowes/echomuse-controller:latest` is
**multi-arch: linux/amd64 + linux/arm64** (arm64 shipped 2026-08-13).
No `platform:` line is needed — compose resolves the right one.

`network_mode: host` is **required**: the controller advertises over mDNS
and opens one ESPHome TCP port per device, which does not work behind
Docker's port mapping. It also means the container shares the host's
network namespace, so the dashboard is on the host's LAN IP.

Deploy exactly as upstream documents:

```bash
mkdir echomuse && cd echomuse
curl -O https://raw.githubusercontent.com/wilbowes/EchoMuse/main/controller/docker-compose.deploy.yml
curl -o .env https://raw.githubusercontent.com/wilbowes/EchoMuse/main/controller/.env.example
# edit .env — set SERVER_IP to this machine's LAN IP
docker compose -f docker-compose.deploy.yml up -d
```

**`.env` — only two values need changing:**

| Key | Default | Set to |
|---|---|---|
| `SERVER_IP` | `10.x.y.z` (placeholder) | the host's LAN IP — this is the address every device is told to dial, and the controller refuses to start rather than advertise a guessed one |
| `ESPHOME_PROJECT_VERSION` | `2.0.0` (placeholder) | the controller release tag, e.g. `2.27.0` |

`ESPHOME_PROJECT_VERSION` matters because the env var **overrides** the
version the image bakes in (`em_esphome.py`: `os.environ.get(
"ESPHOME_PROJECT_VERSION", _CONTROLLER_VERSION)`), and HA displays it as
the satellite's version. Left at `2.0.0`, HA shows 2.0.0 forever and
updates misreport.

Leave `DEVICE_APPROVAL=strict` on a first run: nothing joins the voice
network without the owner's approval.

## Discovery requirements

- The Echo finds the controller over **mDNS, which does not cross subnets
  or VLANs**. Controller and device must share a LAN segment, or the
  device needs a static controller endpoint configured.
- With a static endpoint the firewall still has to let the Echo open TCP
  8767 and 8770. An mDNS relay between VLANs was not enough for one owner
  (#827).
- Set Home Assistant's local URL to its **IP address, not a `.local`
  name** — the controller fetches reply audio from that URL.

## Other host requirements

- `ffmpeg` on PATH (playback, TTS decoding, timers) — the image ships it.
- `git` (version reporting) — the image ships it.
- `DB_PATH` defaults to `/app/data/echomuse.db`; the compose file mounts
  `./data:/app/data`, so the database, device-link certificates and
  recordings persist across upgrades. **Back up that directory** — it
  holds the devices, users, config, activity history and the TLS CA.
- The device-link CA is generated on first start under `data/tls/`. Move
  devices to a new controller only by copying that directory across, or
  they will refuse to trust it.
- The **dashboard/API port is plain HTTP**; there is no HTTPS option for
  it. Only the device link is encrypted.

## Home Assistant integration

After the device is approved in the dashboard, add one **ESPHome**
integration per device: the controller's IP and that device's port
(16001 for the first, 16002 for the second, and so on). Then assign the
device to an Assist pipeline. The controller exposes the device as
`<name> Voice Assistant`, model `Echo Dot Gen 2 (biscuit)`.
