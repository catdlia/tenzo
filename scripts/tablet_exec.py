#!/usr/bin/env python3
"""
Tablet Remote Command Runner over tmate SSH
"""
import sys
import pty
import os
import select
import time
import fcntl
import struct
import termios

SSH_CMD = ["ssh", "-tt", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=15", "UFFn6rRTjNmeP8D5u9x99L5Yn@sfo2.tmate.io"]

def run_on_tablet(command, timeout=60):
    master, slave = pty.openpty()
    winsize = struct.pack('HHHH', 24, 120, 0, 0)
    fcntl.ioctl(master, termios.TIOCSWINSZ, winsize)

    pid = os.fork()
    if pid == 0:
        os.close(master)
        os.environ['TERM'] = 'xterm-256color'
        os.dup2(slave, 0)
        os.dup2(slave, 1)
        os.dup2(slave, 2)
        os.close(slave)
        os.execlp(SSH_CMD[0], *SSH_CMD)

    os.close(slave)
    
    # Wait for tmate connection
    time.sleep(2.0)
    # Clear screen and reset line
    os.write(master, b'\x03\x03\r\n') 
    time.sleep(0.5)
    
    delim_start = "===T_START==="
    delim_end = "===T_END==="
    
    full_cmd = f"echo {delim_start} && {command} && echo {delim_end}\r\n"
    os.write(master, full_cmd.encode('utf-8'))
    
    start_time = time.time()
    buf = b""
    
    while time.time() - start_time < timeout:
        r, _, _ = select.select([master], [], [], 0.5)
        if not r:
            continue
        try:
            chunk = os.read(master, 4096)
            if not chunk:
                break
            buf += chunk
            text = buf.decode('utf-8', errors='ignore')
            if delim_end in text:
                break
        except OSError:
            break

    try:
        os.close(master)
    except:
        pass

    text = buf.decode('utf-8', errors='ignore')
    if delim_start in text:
        content = text.split(delim_start, 1)[1]
        if delim_end in content:
            content = content.split(delim_end, 1)[0]
        return content.strip()
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: tablet_exec.py <command>")
        sys.exit(1)
    cmd = " ".join(sys.argv[1:])
    out = run_on_tablet(cmd, timeout=120)
    print(out)
