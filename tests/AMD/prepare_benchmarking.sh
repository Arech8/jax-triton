#!/usr/bin/bash

if [ "$0" = "$BASH_SOURCE" ]; then
    echo "Error: Script must be sourced"
    exit 1
fi

function setup_benchmarking() {
    # Implemented here is gfx950 specific and likely incomplete, as it helps less than
    # desirable. However, it's still better than nothing

    # rocm-smi --setfan 100% # fan control isn't supported on gfx950
    rocm-smi --setperfdeterminism 0
    rocm-smi --setpoweroverdrive 900  # 1400W is gfx950 TDP. Note, setting too low would slowdown gluon!

    #cpupower frequency-set -g performance
    #sysctl kernel.numa_balancing=0
}

set -x

export ROCR_VISIBLE_DEVICES=1

setup_benchmarking

rocm-smi --showperflevel

set +x

function teardown_benchmarking() {
    set -x

    unset ROCR_VISIBLE_DEVICES

    rocm-smi --resetperfdeterminism --resetpoweroverdrive --resetfans --setperflevel auto
    rocm-smi --showperflevel

    set +x
}

echo "Benchmarking setup complete. To revert, run: teardown_benchmarking. To setup again, run: setup_benchmarking"