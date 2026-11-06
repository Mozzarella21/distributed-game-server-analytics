#!/usr/bin/env python3
import sys

current_key = None
total_latency = 0.0
count = 0

for line in sys.stdin:
    key, latency, n = line.strip().split("\t")

    if current_key is not None and key != current_key:
        print(f"{current_key}\t{total_latency/count:.2f}\t{count}")
        total_latency = 0.0
        count = 0

    current_key = key
    total_latency += float(latency)
    count += int(n)

if current_key is not None: 
    print(f"{current_key}\t{total_latency/count:.2f}\t{count}")
