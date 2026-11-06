#!/usr/bin/env python3
import sys
import csv

for row in csv.reader(sys.stdin):
    if len(row) != 11:
        continue

    try:
        server_id = row[1]
        region = row[2]
        latency = float(row[6])
        print(f"{server_id},{region}\t{latency}\t1")
    except ValueError:
        continue
