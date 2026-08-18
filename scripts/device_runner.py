#!/usr/bin/env python3
"""
Multi-Device Automated Remote Command Runner for Tenzo
Supports Phone (Termux/tmate), Tablet (Termux/tmate), Laptop (Host/Docker)
"""
import sys
import os
import pty
import select
import time
import fcntl
import struct
import termios
import subprocess

TARGETS = {
    "phone": "qPzEE7SMDX3c7h4hn7aeRezKz@nyc1.tmate.io",
    "tablet": "hhWa46e3Syw4fGqxXJpe3v9c8@nyc1.tmate.io",
}

def exec_remote(target_name, command, timeout=120):
    ssh_target = TARGETS.get(target_name)
    if not ssh_target:
        raise ValueError(f"Unknown target: {target_name}")

    master, slave = pty.openpty()
    winsize = struct.pack('HHHH', 24, 120, 0, 0)
    fcntl.ioctl(master, termios.TIOCSWINSZ, winsize)

    pid = os.fork()
    if pid == 0:
        os.close(master)
        os.environ['TERM'] = 'xterm-256color'
        os.dup2(slave, 0); os.dup2(slave, 1); os.dup2(slave, 2)
        os.close(slave)
        os.execlp('ssh', 'ssh', '-tt', '-o', 'StrictHostKeyChecking=no', '-o', 'ConnectTimeout=10', ssh_target)

    os.close(slave)
    time.sleep(2.0)
    # Dismiss tmate prompt if any
    os.write(master, b'q\r\n\x03\r\n')
    time.sleep(0.5)

    import base64
    delim_start = f"===START_{int(time.time()*1000)}==="
    delim_end = f"===END_{int(time.time()*1000)}==="
    
    script_content = f"echo '{delim_start}'\n{command}\necho '{delim_end}'\n"
    b64 = base64.b64encode(script_content.encode('utf-8')).decode('ascii')
    
    full_cmd = f"echo {b64} | base64 -d | bash\r\n"
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
        # Clean terminal output
        lines = []
        for l in content.split('\n'):
            cleaned = l.replace('\r', '').strip()
            if cleaned and not cleaned.startswith('===START_') and not cleaned.startswith('===END_'):
                lines.append(cleaned)
        return "\n".join(lines)
    return text

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: device_runner.py <phone|tablet> <command...>")
        sys.exit(1)
    target = sys.argv[1]
    cmd = " ".join(sys.argv[2:])
    print(exec_remote(target, cmd))
