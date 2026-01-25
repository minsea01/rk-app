#!/usr/bin/env python3
"""
CityPersons 标注文件下载器
支持多个镜像源，适配国内网络环境
"""

import os
import sys
import urllib.request
import hashlib
from pathlib import Path

# 下载源列表（按优先级排序）
SOURCES = {
    "anno_train.mat": [
        # Gitee 镜像 (国内快)
        "https://gitee.com/mirrors_cvgroup-njust/CityPersons/raw/master/annotations/anno_train.mat",
        # GitHub 直连
        "https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/anno_train.mat",
        # GitHub 镜像 (ghproxy)
        "https://ghproxy.com/https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/anno_train.mat",
        # 备用 GitHub 仓库
        "https://github.com/CharlesShang/Detectron-OHEM/raw/master/data/citypersons/annotations/anno_train.mat",
        # jsdelivr CDN
        "https://cdn.jsdelivr.net/gh/cvgroup-njust/CityPersons@master/annotations/anno_train.mat",
    ],
    "anno_val.mat": [
        "https://gitee.com/mirrors_cvgroup-njust/CityPersons/raw/master/annotations/anno_val.mat",
        "https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/anno_val.mat",
        "https://ghproxy.com/https://github.com/cvgroup-njust/CityPersons/raw/master/annotations/anno_val.mat",
        "https://github.com/CharlesShang/Detectron-OHEM/raw/master/data/citypersons/annotations/anno_val.mat",
        "https://cdn.jsdelivr.net/gh/cvgroup-njust/CityPersons@master/annotations/anno_val.mat",
    ],
}

# 预期文件大小 (字节，用于验证) - CityPersons 标注文件较小
EXPECTED_SIZES = {
    "anno_train.mat": (100000, 5000000),   # 100KB-5MB (实际约 378KB)
    "anno_val.mat": (30000, 1000000),       # 30KB-1MB (实际约 75KB)
}


def download_file(url: str, dest: Path, timeout: int = 30) -> bool:
    """下载文件，返回是否成功"""
    try:
        print(f"    尝试: {url[:60]}...")

        # 设置 User-Agent 避免被拒绝
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )

        with urllib.request.urlopen(req, timeout=timeout) as response:
            data = response.read()

            # 写入文件
            with open(dest, "wb") as f:
                f.write(data)

            print(f"    成功! 大小: {len(data)/1024:.1f} KB")
            return True

    except Exception as e:
        print(f"    失败: {str(e)[:50]}")
        return False


def verify_file(filepath: Path, filename: str) -> bool:
    """验证文件完整性"""
    if not filepath.exists():
        return False

    size = filepath.stat().st_size
    min_size, max_size = EXPECTED_SIZES.get(filename, (1000, 10000000))

    if size < min_size or size > max_size:
        print(f"  警告: {filename} 大小异常 ({size} bytes)")
        return False

    # 尝试用 scipy 加载验证
    try:
        import scipy.io as sio
        data = sio.loadmat(str(filepath))
        # 检查是否有预期的 key
        expected_keys = [f"anno_{filename.replace('anno_', '').replace('.mat', '')}"]
        for key in expected_keys:
            if key in data or f"{key}_aligned" in data:
                return True
        print(f"  警告: {filename} 格式不正确")
        return False
    except Exception as e:
        print(f"  警告: 无法验证 {filename}: {e}")
        # 如果 scipy 未安装，只检查大小
        return True


def main():
    output_dir = Path(__file__).parent

    print("=" * 60)
    print("  CityPersons 标注文件下载器")
    print("  适配国内网络环境，多镜像源自动切换")
    print("=" * 60)
    print()

    success_count = 0

    for filename, urls in SOURCES.items():
        dest = output_dir / filename

        print(f"[{filename}]")

        # 检查是否已存在
        if dest.exists() and verify_file(dest, filename):
            print(f"  已存在且有效，跳过")
            success_count += 1
            continue

        # 尝试各个源
        downloaded = False
        for url in urls:
            if download_file(url, dest):
                if verify_file(dest, filename):
                    downloaded = True
                    success_count += 1
                    break
                else:
                    dest.unlink()  # 删除无效文件

        if not downloaded:
            print(f"  所有源下载失败!")
            print()
            print("  手动下载方法:")
            print("  1. 访问 https://github.com/cvgroup-njust/CityPersons")
            print("  2. 下载 annotations/anno_train.mat 和 anno_val.mat")
            print(f"  3. 放到 {output_dir}/")

        print()

    print("=" * 60)
    if success_count == len(SOURCES):
        print("  全部下载成功!")
        print(f"  文件位置: {output_dir}")
    else:
        print(f"  下载完成: {success_count}/{len(SOURCES)}")
        sys.exit(1)
    print("=" * 60)


if __name__ == "__main__":
    main()
