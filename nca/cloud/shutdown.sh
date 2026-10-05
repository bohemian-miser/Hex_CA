#!/bin/bash
# The VM's shutdown-script (launch.sh passes it): one last quick copy of the run files and status.json to the
# bucket. A Spot preemption gives ~30 s; startup.sh's own TERM trap does the same, so either is enough.
[ -f /opt/hexca/main.sh ] && timeout 25 bash /opt/hexca/main.sh sync
exit 0
