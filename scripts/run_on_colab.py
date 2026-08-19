import os
import tarfile
import subprocess
import sys

# Extract payload
with tarfile.open('/content/payload.tar', 'r') as f:
    f.extractall('/content/')

# Execute the training script
result = subprocess.run(
    [sys.executable, 'scripts/colab_graphsage_smoke_training.py'],
    cwd='/content/',
    capture_output=True,
    text=True
)

print(result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

if result.returncode != 0:
    sys.exit(result.returncode)
