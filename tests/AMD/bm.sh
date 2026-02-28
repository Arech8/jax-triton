#!/usr/bin/bash

DUMP_DIR="$PWD"
FILES_PFX="${DUMP_DIR}/bmr"
CUST_PRMS="--iters=100 --reps=10 --dump_pfx=${FILES_PFX}"

N_TRIES=3

OLD="${FILES_PFX}*.svg"
OLD_NPY="${FILES_PFX}*.npy"

set -x
rm $OLD
rm $OLD_NPY
[ ! -d $DUMP_DIR ] && mkdir -p $DUMP_DIR

# custom vs normal runner on gemms only
for (( i=0 ; i < $N_TRIES ; i++ )) do
    echo "iteration $i"
    python benchmarks.py ${CUST_PRMS} gemm_afp4wfp4 --custom_runner
    python benchmarks.py ${CUST_PRMS} gemm_afp4wfp4 --no-custom_runner
done

# single gemm with and without inputs randomization
for (( i=0 ; i < $N_TRIES ; i++ )) do
    echo "iteration $i"
    python benchmarks.py ${CUST_PRMS} --single_gemm gemm_afp4wfp4 --random_inputs
    python benchmarks.py ${CUST_PRMS} --single_gemm gemm_afp4wfp4 --no-random_inputs
done

# startup bms with and without randomized iterations
for (( i=0 ; i < $N_TRIES ; i++ )) do
    echo "iteration $i"
    python benchmarks.py ${CUST_PRMS} startup --randomize_iterations --random_inputs
    python benchmarks.py ${CUST_PRMS} startup --no-randomize_iterations --random_inputs
    python benchmarks.py ${CUST_PRMS} startup --no-randomize_iterations --no-random_inputs
done

# all bms with and without randomized iterations, and all without random iters and inputs
for (( i=0 ; i < $N_TRIES ; i++ )) do
    echo "iteration $i"
    python benchmarks.py ${CUST_PRMS} --randomize_iterations --random_inputs
    python benchmarks.py ${CUST_PRMS} --no-randomize_iterations --random_inputs
    python benchmarks.py ${CUST_PRMS} --no-randomize_iterations --no-random_inputs
done

set +x