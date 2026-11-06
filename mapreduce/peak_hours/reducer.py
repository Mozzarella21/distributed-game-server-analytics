
#!/usr/bin/env python3
import sys

current_hour = None
total_events = 0

for line in sys.stdin:
    parts = line.strip().split("\t")

    if len(parts) != 2:
        continue

    hour, count = parts

    if current_hour is not None and hour != current_hour:
        print(f"{current_hour}\t{total_events}")
        total_events = 0

    current_hour = hour
    total_events += int(count)

if current_hour is not None:
    print(f"{current_hour}\t{total_events}")
