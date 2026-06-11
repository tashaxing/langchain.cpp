---
name: sysinfo
description: Get system information like CPU core count, memory usage, and disk space
type: script
template: "Collect system info with filter={filter}"
parameters:
  type: object
  properties:
    filter:
      type: string
      description: What info to collect
      enum:
        - cpu
        - memory
        - disk
        - all
  required:
    - filter
script_dir: scripts
script_name: run
---

# System Info Skill

Collects system information such as CPU, memory, and disk usage by executing platform-specific scripts.
