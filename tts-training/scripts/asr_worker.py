"""Subprocess supervision; GPU reservation lives as long as the worker."""
import argparse
import fcntl
import os
from pathlib import Path
import signal
import subprocess
import sys
import stat
import time
from asr_runtime import atomic_json, heartbeat, read_json


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=5)
    except ProcessLookupError:
        pass
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=5)
    finally:
        # A successful leader can still leave a background descendant behind.
        try:os.killpg(process.pid,signal.SIGKILL)
        except ProcessLookupError:pass


def memory_sample():
    try:
        relative = next(line.split(':', 2)[2].strip() for line in Path('/proc/self/cgroup').read_text().splitlines()
                        if line.startswith('0::'))
        directory = Path('/sys/fs/cgroup') / relative.lstrip('/')
        sample = {name: (directory/name).read_text().strip() for name in
                  ['memory.current', 'memory.peak', 'memory.swap.current', 'memory.events','memory.stat']}
        return sample
    except (OSError, StopIteration):
        return {}


def device_memory():
    binary = '/usr/lib/wsl/lib/nvidia-smi' if Path('/usr/lib/wsl/lib/nvidia-smi').exists() else 'nvidia-smi'
    output = subprocess.check_output([binary, '--query-gpu=memory.free,memory.total',
                                      '--format=csv,noheader,nounits'], timeout=10, text=True)
    free, total = output.splitlines()[0].split(',')
    return dict(free_mib=int(free.strip()), total_mib=int(total.strip()))


def process_identity(pid):
    try:
        fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
        return None if fields[0]=='Z' else fields[19]
    except (OSError,IndexError):
        return None


def validate_inherited_lock(fd, path, controller_pid, controller_start_time):
    """Accept only an existing exclusive flock on this exact reservation file."""
    if controller_pid <= 0 or not controller_start_time or process_identity(controller_pid) != controller_start_time:
        raise RuntimeError('invalid_lock_controller')
    inherited, expected = os.fstat(fd), path.stat()
    if not stat.S_ISREG(inherited.st_mode) or (inherited.st_dev, inherited.st_ino) != (expected.st_dev, expected.st_ino):
        raise RuntimeError('invalid_inherited_lock_file')
    # flock belongs to the shared open-file description. Do not flock a separately
    # opened handle, or unlock this one: either would break pipeline ownership.
    info = Path(f'/proc/self/fdinfo/{fd}').read_text()
    if not any(line.startswith('lock:') and line.split()[2:5] == ['FLOCK', 'ADVISORY', 'WRITE'] for line in info.splitlines()):
        raise RuntimeError('inherited_lock_not_exclusive')


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--run-id', required=True)
    parser.add_argument('--stage', required=True)
    parser.add_argument('--profile', choices=['cpu', 'asr'], required=True)
    parser.add_argument('--runtime', type=float, default=0)
    parser.add_argument('--load-timeout', type=float, default=0)
    parser.add_argument('--idle-timeout', type=float, default=0)
    parser.add_argument('--min-ram-mib', type=int, default=0)
    parser.add_argument('--min-vram-mib', type=int, default=0)
    parser.add_argument('--controller-pid',type=int,default=0)
    parser.add_argument('--controller-start-time',default='')
    parser.add_argument('--inherited-lock-fd',type=int,default=None)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command:
        parser.error('a child command is required')
    directory = args.root / 'reports/asr-runs' / args.run_id
    directory.mkdir(parents=True, exist_ok=True)
    status_path = directory / (args.stage + '.json')
    heartbeat_path = directory / (args.stage + '-heartbeat.json')
    state = dict(schema=2, run_id=args.run_id, stage=args.stage, profile=args.profile,
                 status='starting', started=time.time(), gpu_model_loaded_by_supervisor=False)
    atomic_json(status_path, state)
    lock_path = args.root / '.cache/training-launch.lock'
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    process = None
    requested = []
    def cancel(signum, frame):
        requested.append(signum)
    for signum in [signal.SIGTERM, signal.SIGINT]:
        signal.signal(signum, cancel)
    with lock_path.open('a') as reservation:
        try:
            if args.inherited_lock_fd is None:
                try:
                    fcntl.flock(reservation, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    raise RuntimeError('resource_busy')
                reservation_fd = reservation.fileno()
                state['lock_mode'] = 'standalone'
            else:
                validate_inherited_lock(args.inherited_lock_fd, lock_path,
                                        args.controller_pid, args.controller_start_time)
                reservation_fd = args.inherited_lock_fd
                state['lock_mode'] = 'inherited'
            available = int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                                 if line.startswith('MemAvailable:'))) / 1024
            if args.min_ram_mib and available < args.min_ram_mib:
                raise RuntimeError('insufficient_system_memory')
            if args.profile == 'asr':
                state['gpu_before'] = device_memory()
                if args.min_vram_mib and state['gpu_before']['free_mib'] < args.min_vram_mib:
                    raise RuntimeError('insufficient_gpu_memory')
            env = os.environ.copy()
            env.update(ASR_RUN_ID=args.run_id, ASR_HEARTBEAT_PATH=str(heartbeat_path))
            env['PYTHONPATH'] = ''
            started = time.monotonic()
            process = subprocess.Popen(command, cwd=args.root, env=env, start_new_session=True,
                                       pass_fds=(reservation_fd,))
            state.update(status='running', child_pid=process.pid)
            atomic_json(status_path, state)
            loading_since = None
            last_record = 0.0
            while process.poll() is None:
                now = time.monotonic()
                beat = read_json(heartbeat_path, {})
                valid = beat.get('run_id') == args.run_id
                last_progress = beat.get('monotonic', started) if valid else started
                if valid and beat.get('phase') == 'model_loading':
                    loading_since = loading_since or last_progress
                elif valid:
                    loading_since = None
                if requested:
                    raise RuntimeError('cancelled')
                if args.controller_pid and process_identity(args.controller_pid)!=args.controller_start_time:
                    raise RuntimeError('controller_disappeared')
                if args.runtime and now-started > args.runtime:
                    raise RuntimeError('runtime_timeout')
                if args.load_timeout and loading_since is not None and now-loading_since > args.load_timeout:
                    raise RuntimeError('loader_timeout')
                if args.idle_timeout and now-last_progress > args.idle_timeout:
                    raise RuntimeError('no_progress_timeout')
                if now-last_record > 5:
                    state.update(heartbeat=beat if valid else None, memory=memory_sample(),
                                 elapsed_seconds=now-started)
                    atomic_json(status_path, state)
                    last_record = now
                time.sleep(min(.25, args.idle_timeout/4) if args.idle_timeout else .25)
            if requested:
                raise RuntimeError('cancelled')
            if process.returncode:
                state['child_exit_code'] = process.returncode
                raise RuntimeError('child_failed')
            state.update(status='success', complete=True, child_exit_code=0)
            return_code = 0
        except (RuntimeError, OSError, subprocess.SubprocessError) as error:
            state.update(status='failed', complete=False, failure_kind=str(error))
            return_code = 1
        finally:
            if process is not None:
                terminate(process)
            state.update(finished=time.time(), memory=memory_sample(), worker_stopped=True)
            atomic_json(status_path, state)
    return return_code


if __name__ == '__main__':
    sys.exit(main())
