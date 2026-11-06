
#!/usr/bin/env python3
import sys
import csv

for row in csv.reader(sys.stdin):
    if len(row) != 11:
        continue

    server_id = row[1]
    event_type = row[5].strip().upper()

    disconnect = 1 if event_type == "DISCONNECT" else 0

    print(f"{server_id}\t1\t{disconnect}")
