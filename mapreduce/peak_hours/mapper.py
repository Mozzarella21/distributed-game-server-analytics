
#!/usr/bin/env python3
import sys
import csv

for row in csv.reader(sys.stdin):
    if len(row) != 11:
        continue

    try:
        timestamp = row[0].strip()
        hour = int(timestamp[11:13])

        if not 0 <= hour <= 23:
            continue

        print(f"{hour:02d}\t1")

    except (ValueError, IndexError):
        continue
