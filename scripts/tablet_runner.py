#!/usr/bin/env python3
"""
Tablet SSH Interactive Command Executor
"""
import sys
import pty
import os
import select
import time
import fcntl
import struct
import termios

SSH_TARGET = "UFFn6rRTjNmeP8D5u9x99L5Yn@sfo2.tmate.io"

def exec_tablet(cmd_str, timeout=120):
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
        os.execlp('ssh', 'ssh', '-tt', '-o', 'StrictHostKeyChecking=no', SSH_TARGET)

    os.close(slave)
    time.sleep(2.0)
    os.write(master, b'\x03\r\n')
    time.sleep(0.5)

    import base64
    tag_start = "###CMD_OUTPUT_START###"
    tag_end = "###CMD_OUTPUT_END###"
    
    script = f"echo '{tag_start}'; {cmd_str}; echo '{tag_end}'\n"
    b64_script = base64.b64encode(script.encode('utf-8')).decode('ascii')
    
    os.write(master, f"echo {b64_script} | base64 -d | bash\r\n".encode('utf-8'))

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
            if tag_end in text:
                break
        except OSError:
            break

    try:
        os.close(master)
    except:
        pass

    text = buf.decode('utf-8', errors='ignore')
    if tag_start in text:
        content = text.split(tag_start, 1)[1]
        if tag_end in content:
            content = content.split(tag_end, 1)[0]
        return content.strip()
    return text

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: tablet_runner.py <bash_command>")
        sys.exit(1)
    print(exec_tablet(" ".join(sys.argv[1:])))
