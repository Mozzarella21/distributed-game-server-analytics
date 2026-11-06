#!/usr/bin/env python3
import sys

current_server = None
total = 0

for line in sys.stdin:
    server, count = line.strip().split("\t")
    count = int(count)

    if server == current_server:
        total += count
    else:
        if current_server is not None:
            print(f"{current_server}\t{total}")

        current_server = server
        total = count

if current_server is not None:
    print(f"{current_server}\t{total}")
