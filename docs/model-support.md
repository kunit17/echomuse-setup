# Model support

## How to identify the hardware

The model number is printed on the label on the bottom of the device and is
shown in the Alexa app under the device's settings. The codename matters as
much as the model number — several models share a generation but not a
codename, and the codename is what the unlock exploits target.

## EchoMuse support

EchoMuse's README states the supported set plainly. Anything not listed is
not supported, and no amount of configuration makes it so.

| Model | Number | Codename | EchoMuse support |
|---|---|---|---|
| Echo Dot 2nd gen (2016) | RS03QR | biscuit | **Supported** — the flagship target, best-tested path |
| Echo 2nd gen (2017) | XC56PY | radar | **Supported** |
| Echo Dot 3rd gen (2018) | D9N29T | donut | **Early Access only** — firmware support exists but the wizard/emOS path is still being finished; treat as a pilot, not a production install |
| Echo Dot 3rd gen refresh (2019) | C78MP8 | crumpet | **Not supported** |
| Echo Dot 3rd gen with clock (2019) | 36EBT3 | doebrite | **Not supported** |
| Echo Dot 4th gen (2020) | B7W64E | brownie | **Not supported** — and no public unlock exploit exists |
| Echo Dot 4th gen with clock (2020) | B7W644 | ganache | **Not supported** |
| Echo Dot 5th gen (2022) | C2N6L4 | cannoli | **Not supported** by EchoMuse |

The 3rd gen is the trap: an "Early Access" build exists and the firmware
carries a board definition for it, which reads as support. It is not the
same as a released, supported target.

## When EchoMuse does not support the model

The alternative project **echo-dot-assist** covers more hardware because it
keeps Amazon's own firmware and wake-word engine and replaces only the Alexa
client. It covers Echo Dot 3rd gen, Echo Dot 5th gen, the Echo Shows, the
Echo Input and more.

Trade-offs when choosing it:

- It is self-described as unaudited and explicitly "your risk" — it writes
  the system partition and can brick the device.
- Its install path is per-model and partly manual. Models marked as
  working-by-hand need the case opened and USB soldered to test pads, which
  is a different order of physical work than a USB cable.
- The unlock is done by hand per that model's own forum thread, not by a
  single command.

**Prefer EchoMuse whenever the model is on its supported list.** Only move to
the alternative when the model is definitively unsupported, and present the
trade-off before starting, because the risk profile is materially higher.

## Model numbers are not generation numbers

The 3rd gen refresh and the 3rd gen with clock are different hardware from
the original 3rd gen and are separately unsupported. A 2019 "Echo Dot 3rd
gen" bought second-hand may be either. Read the exact model number off the
label rather than trusting the listing or the generation name.

## Ecosystem

This is a MediaTek-based family. The unlock exploits are bootrom-level and
were written per-chipset by the people who ported them; the fork that
matters is the one matching the device's codename, and its own forum thread
is the authority for that device. The general project's README links each
codename to its thread — go there for device-specific recovery notes.
