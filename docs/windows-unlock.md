# firebreak — native Windows unlock of an Echo Dot 2nd gen

The unlock runs on the machine the Dot is plugged into. Windows is a
first-class documented host: firebreak's README has its own `### Windows`
section and warns that it is tested *only* from an administrator
PowerShell, for both setup and run.

## What firebreak does to the device (the destructive part)

It is not a soft operation. Per firebreak's `docs/rooting.md`:

- runs a **bootrom exploit** (amonet) against the MediaTek SoC
- **rewrites the preloader, LK and TrustZone**, and the partition table
- **formats userdata** — Wi-Fi credentials and Alexa registration are gone
- installs TWRP

Recovery: a failed unlock leaves the Dot in TWRP, which is recoverable. A
Dot that will not boot at all is also recoverable, but *"the easy routes
are gone; getting back in means opening the case and shorting a pin on the
board"*. **Never power-cycle a Dot that will not boot** — the bootloader
counts boot attempts per slot and stops booting once both slots are
exhausted.

## Prerequisites (from firebreak's README, verbatim)

```powershell
winget install Google.PlatformTools
winget install astral-sh.uv
```

Then the fastboot and MTK CDC-ACM drivers, libusb, and UsbDk. **UsbDk is a
kernel-mode USB filter driver** — installed via MSI. Do not install it
silently (`/qn`); show the UI. The driver CABs come from
`catalog.s.download.windowsupdate.com` (Microsoft's update catalog) and
libusb from the libusb GitHub release.

`libusb-1.0.dll` goes on the **user** PATH, not the system PATH.

**Windows CAB extraction:** the driver files are `.cab`, and
`Expand-Archive` cannot read `.cab` — only `.zip`. Use `expand.exe -F:*`,
which is what firebreak's own instructions use.

## Running it

```powershell
uvx --managed-python firebreak@latest unlock
```

`--managed-python` is required: firebreak stops rather than guess when uv
would use a Microsoft Store Python, because adb and fastboot cannot see
that Python's files.

The run is **interactive and physical** — firebreak prints the button
gesture it wants (mute + volume-down for a stock Dot) and waits. It polls
the Dot every 2 seconds, so a stopped run can be resumed by re-running
it. Expect ~10 minutes.

Targets on a Dot 2: `unlock` (default) leaves it on FireOS 6, which is
what **emOS** requires. `unlock fireos5` leaves it on FireOS 5.5.5.4 with
TWRP for the FireOS-with-root path. The two are **not equally
reversible** — read the project's rooting guide before choosing.

## Before the unlock

- **Never set the Dot up in the Alexa app first.** A stock Echo on Wi-Fi
  can take an OTA that moves it to a bootloader the unlock path does not
  handle.
- The data cable must carry data, not just power.
- If a USB port browns out, the Dot boot-loops. Its own adapter supplies
  9 W; a computer's USB 2.0 port supplies only 2.5 W.
- `adb kill-server` before the run — a live adb server holds the device
  and the browser wizard later fails with "device in use".

## After the unlock, the wizard

The provisioning wizard runs in Chrome or Edge on the **same machine as
the USB cable**, and talks to the Dot over **WebUSB**. See
`references/controller-hosting.md` for the secure-context constraint when
the controller is on another host.
