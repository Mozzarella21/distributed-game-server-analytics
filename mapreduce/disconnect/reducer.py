
#!/usr/bin/env python3
import sys

current_server = None
total_events = 0
total_disconnects = 0

def output_result():
    if current_server is not None and total_events > 0:
        rate = (total_disconnects / total_events) * 100
        print(
            f"{current_server}\t"
            f"{total_events}\t"
            f"{total_disconnects}\t"
            f"{rate:.2f}"
        )

for line in sys.stdin:
    parts = line.strip().split("\t")

    if len(parts) != 3:
        continue

    server_id, events, disconnects = parts

    if current_server is not None and server_id != current_server:
        output_result()
        total_events = 0
        total_disconnects = 0

    current_server = server_id
    total_events += int(events)
    total_disconnects += int(disconnects)

output_result()
