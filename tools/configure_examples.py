#!/usr/bin/env python3
"""Personalize an unshared local clone. Dry-run by default; never push the result."""
import argparse,ipaddress
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--wsl-user',required=True)
p.add_argument('--pi-user',required=True)
p.add_argument('--windows-user',required=True)
p.add_argument('--pi-host',required=True)
p.add_argument('--desktop-host',required=True)
p.add_argument('--apply',action='store_true')
a=p.parse_args()
for value in [a.wsl_user,a.pi_user,a.windows_user]:
    if not value or any(c in value for c in '/\\\n\r'):p.error('Invalid account name')
ipaddress.ip_address(a.pi_host);ipaddress.ip_address(a.desktop_host)
root=Path(__file__).resolve().parents[1]
mapping={'voiceuser':a.wsl_user,'piuser':a.pi_user,'DesktopUser':a.windows_user,
         '192.0.2.10':a.pi_host,'192.0.2.20':a.desktop_host}
count=0
for f in root.rglob('*'):
    if not f.is_file() or '.git' in f.parts or f==Path(__file__).resolve():continue
    try:text=f.read_text()
    except (UnicodeError,OSError):continue
    changed=text
    for old,new in mapping.items():changed=changed.replace(old,new)
    if changed!=text:
        count+=1
        if a.apply:f.write_text(changed)
print(('Updated' if a.apply else 'Would update'),count,'files. Keep the personalized clone private.')
