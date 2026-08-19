import subprocess

p = subprocess.Popen(
    ['/Users/acelyayildiz/Library/Python/3.12/bin/colab', 'console', '-s', '76ce3a'],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)
out, _ = p.communicate(input="cd /content\ntar -xvf payload.tar\npython3 scripts/colab_graphsage_smoke_training.py\nexit\n")
print(out)
