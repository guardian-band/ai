import pty
import os
import time

pid, fd = pty.fork()
if pid == 0:
    # Child process
    os.execv('/Users/acelyayildiz/Library/Python/3.12/bin/colab', ['colab', 'console', '-s', '76ce3a'])
else:
    # Parent process
    time.sleep(3)
    os.write(fd, b"cd /content && tar -xvf payload.tar\n")
    time.sleep(2)
    os.write(fd, b"python3 scripts/colab_graphsage_smoke_training.py\n")
    
    output = b""
    start_time = time.time()
    
    # Read output non-blocking for up to 60 seconds
    import select
    while time.time() - start_time < 60:
        r, w, x = select.select([fd], [], [], 1.0)
        if fd in r:
            try:
                data = os.read(fd, 1024)
                if not data:
                    break
                output += data
                print(data.decode('utf-8', errors='replace'), end='', flush=True)
                
                if b"Smoke training successfully completed." in data:
                    break
            except OSError:
                break
                
    os.write(fd, b"exit\n")
