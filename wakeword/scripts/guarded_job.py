#!/usr/bin/env python3
"""Run a trainer preparation job while watching real Windows C: headroom."""
import argparse,json,os,signal,subprocess,time,shutil
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--name',required=True)
p.add_argument('--minimum-free-gib',type=float,default=10)
p.add_argument('--container')
p.add_argument('command',nargs=argparse.REMAINDER);args=p.parse_args()
command=args.command[1:] if args.command[:1]==['--'] else args.command
if not command:raise SystemExit('Command required')
root=Path(__file__).resolve().parent;log_path=root/'logs'/(args.name+'.log')
status_path=root/'logs'/(args.name+'.json');cancelled=False

def cancel(signum,frame):
 global cancelled
 cancelled=True
signal.signal(signal.SIGINT,cancel);signal.signal(signal.SIGTERM,cancel)
started=time.time();process=None;reason=None
controller_pid=os.getppid()
def identity(pid):
 try:
  fields=Path('/proc/'+str(pid)+'/stat').read_text().rsplit(')',1)[1].split()
  return None if fields[0]=='Z' else fields[19]
 except (OSError,IndexError):return None
controller_identity=identity(controller_pid)
if len(command)>1 and command[1]=='/home/voiceuser/projects/gpt-sovits/scripts/asr_worker.py':
 command[2:2]=['--controller-pid',str(os.getpid()),'--controller-start-time',identity(os.getpid())]
with log_path.open('ab') as log:
 process=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 while process.poll() is None:
  free=shutil.disk_usage('/mnt/c').free
  record=dict(name=args.name,started=started,updated=time.time(),pid=process.pid,
              c_free_gib=free/1024**3,status='running')
  status_path.write_text(json.dumps(record,indent=2)+'\n')
  controller_gone=identity(controller_pid)!=controller_identity
  if cancelled or controller_gone or free<args.minimum_free_gib*1024**3:
   reason='cancelled' if cancelled else 'controller_disappeared' if controller_gone else 'physical_c_space_low'
   if args.container:
    subprocess.run(['docker','stop','--time','10',args.container],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=False)
   elif process.poll() is None:
    os.killpg(process.pid,signal.SIGINT)
   try:process.wait(timeout=45)
   except subprocess.TimeoutExpired:
    os.killpg(process.pid,signal.SIGTERM)
    try:process.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(process.pid,signal.SIGKILL);process.wait()
   break
  time.sleep(2)
 code=process.wait()
 # Stop only this newly installed local Docker engine if a pull hits the
 # low-disk limit; no user containers run in it. Prevent background unpacking.
 if reason=='physical_c_space_low' and command[:2]==['docker','pull']:
  subprocess.run(['systemctl','stop','docker.service','docker.socket'],check=False)
 record=dict(name=args.name,started=started,finished=time.time(),exit_code=code,
             c_free_gib=shutil.disk_usage('/mnt/c').free/1024**3,
             status='success' if code==0 and reason is None else 'stopped',reason=reason)
 status_path.write_text(json.dumps(record,indent=2)+'\n');print(json.dumps(record),flush=True)
raise SystemExit(0 if code==0 and reason is None else 1)
