import pexpect
import sys
import time

child = pexpect.spawn('/Users/acelyayildiz/Library/Python/3.12/bin/colab console -s 76ce3a', encoding='utf-8')
child.expect([r'\$', r'#', pexpect.TIMEOUT], timeout=10)

print("Connected to Colab console.")
child.sendline('cd /content && tar -xvf payload.tar')
child.expect([r'\$', r'#', pexpect.TIMEOUT], timeout=10)
print(child.before)

child.sendline('python3 scripts/colab_graphsage_smoke_training.py')
# Training might take some time, wait longer
child.expect([r'\$', r'#', pexpect.TIMEOUT], timeout=60)
print(child.before)

child.sendline('exit')
child.close()
