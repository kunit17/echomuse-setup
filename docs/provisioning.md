# Provision a supported Echo

1. Confirm exact model using model-support.md and current upstream support.
2. Deploy controller/docker-compose.deploy.yml and create .env from the example.
   Set your controller IP and controller-reported ESPHome version; do not assume
   an old example version describes a newer downloaded image.
3. Review windows-unlock.md, scripts/setup-windows.ps1, 0-check-usb.ps1 and
   2-unlock.ps1. Unlocking can change bootloader/partitions and erase data.
   Follow the matching upstream procedure and physical button timing. Do not
   run a Windows unlock against a device already unlocked on Linux.
4. Connect the Dot by data-capable USB to the machine running Chrome/Edge.
   Open the controller provisioning wizard. WebUSB needs a secure browser
   context: use a localhost tunnel to the controller or properly configured
   HTTPS. The dashboard HTTP port and device-link TLS port are different.
   A narrowly scoped insecure-origin Chrome flag is a documented alternative;
   review its implications and remove it when provisioning is complete.
5. Select the actual Echo in the USB picker. If already in TWRP, reuse recovery
   rather than flashing/unlocking again. Follow the wizard's build checks.
6. Escrow and download YOUR stock boot image. Verify file size/hash, keep an
   offline copy, and follow the wizard's slot/system compatibility checks.
7. Install EchoMuse, prepare wake assets, build emOS, flash/verify and reboot as
   the supported wizard directs. Do not power-cycle during writes.
8. Join your Wi-Fi privately. Approve the newly registered device in the controller.
9. In HA Settings → Devices & services, add the discovered ESPHome device.
   Configure the Assist pipeline, expose a test entity and check a voice turn.

After successful provisioning the Dot needs power and Wi-Fi, not a USB connection
to your desktop. Keep recovery files outside this public repository. Never copy
another Dot's boot image, serial, pairing secret or Wi-Fi settings.
