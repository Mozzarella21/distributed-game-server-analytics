#!/usr/bin/env python3
import sys
import csv

for row in csv.reader(sys.stdin):
    if len(row) != 11:
        continue

    server_id = row[1]
    print(f"{server_id}\t1")
