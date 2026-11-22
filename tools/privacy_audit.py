#!/usr/bin/env python3
"""Reject runtime artifacts and common private identifiers. Also run Gitleaks."""
import ast,ipaddress,json,re,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
errors=[];count=0
binary={'.wav','.mp3','.mp4','.m4a','.flac','.png','.jpg','.onnx','.npz','.safetensors','.pt','.pth','.db','.sqlite','.pem','.key','.vhdx'}
for f in root.rglob('*'):
    if not f.is_file() or '.git' in f.parts or '__pycache__' in f.parts:continue
    name=str(f.relative_to(root));count+=1
    if f.suffix.lower() in binary:errors.append((name,'runtime/binary artifact'));continue
    try:text=f.read_text()
    except UnicodeError:errors.append((name,'non-text payload'));continue
    if f.suffix=='.py':
        try:ast.parse(text,filename=name)
        except SyntaxError:errors.append((name,'Python syntax'))
    if f.suffix=='.json':
        try:json.loads(text)
        except ValueError:errors.append((name,'JSON syntax'))
    address_text='\n'.join(line for line in text.splitlines() if not re.match(r'^[A-Za-z0-9_.-]+==[0-9][^ @]*$', line))
    for value in re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b',address_text):
        try:ip=ipaddress.ip_address(value)
        except ValueError:continue
        if ip.is_loopback or value in {'0.0.0.0','255.255.255.255','100.64.0.0'}:continue
        if value.startswith(('192.0.2.','198.51.100.','203.0.113.')):continue
        if ip.is_private or ip in ipaddress.ip_network('100.64.0.0/10'):
            errors.append((name,'non-example private address'))
    if re.search(r'\bG0[A-Z0-9]{12,18}\b',text):errors.append((name,'device serial'))
    if re.search(r'\b01[A-Z0-9]{24}\b',text,re.I):errors.append((name,'HA runtime identifier'))
    if re.search(r'-----BEGIN (?:OPENSSH |RSA |EC )?PRIVATE KEY-----',text):errors.append((name,'private key'))
for name,reason in sorted(set(errors)):print(name+': '+reason)
print(str(count)+' files audited; '+str(len(set(errors)))+' findings. Secret-value scanning is separate.')
sys.exit(bool(errors))
