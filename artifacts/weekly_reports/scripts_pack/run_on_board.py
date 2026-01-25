#!/usr/bin/env python3
"""
在RK3588板子上远程执行命令
用法: python3 run_on_board.py "命令"
"""
import paramiko
import sys

HOST = "192.168.137.226"
USER = "root"
PASS = "123456"

def run_cmd(cmd):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=30)
    
    print(f">>> {cmd}\n")
    stdin, stdout, stderr = ssh.exec_command(cmd, timeout=120)
    print(stdout.read().decode())
    err = stderr.read().decode()
    if err:
        print(f"[stderr] {err}")
    
    ssh.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python3 run_on_board.py \"命令\"")
        print("示例: python3 run_on_board.py \"bash /root/demo_npu.sh\"")
        sys.exit(1)
    run_cmd(sys.argv[1])
