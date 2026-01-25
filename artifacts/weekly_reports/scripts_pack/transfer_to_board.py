#!/usr/bin/env python3
"""
快速传输文件到RK3588板子
用法: python3 transfer_to_board.py <本地文件> <远程路径>
"""
import paramiko
import sys
import os

HOST = "192.168.137.226"
USER = "root"
PASS = "123456"

def transfer(local_path, remote_path):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(HOST, username=USER, password=PASS, timeout=30)
    
    sftp = ssh.open_sftp()
    file_size = os.path.getsize(local_path)
    print(f"传输: {local_path} -> {HOST}:{remote_path}")
    print(f"大小: {file_size / 1024 / 1024:.1f} MB")
    
    sftp.put(local_path, remote_path)
    print("✅ 传输完成!")
    
    sftp.close()
    ssh.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python3 transfer_to_board.py <本地文件> <远程路径>")
        print("示例: python3 transfer_to_board.py model.rknn /root/rk-app/artifacts/models/")
        sys.exit(1)
    transfer(sys.argv[1], sys.argv[2])
