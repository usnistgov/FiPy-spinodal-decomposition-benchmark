#!/bin/bash

# run_benchmarks.sh
# This script runs each benchmark problem in serial and stores the results
# as a YAML file compatible with the CHiMaD Phase Field schema.
#
# Questions/comments to trevor.keller@nist.gov (Trevor Keller)

# Valid flags are:
# --np X    pass mpirun X ranks (e.g., --np 3 yields mpirun -np 3)
# --short   execute tests .1x default
# --long    execute tests  5x longer  than default
# --extra   execute tests 25x longer  than default
# --clean   delete generated data after test completes

# Set output colors
RED='\033[0;31m'
GRN='\033[0;32m'
WHT='\033[0m' # No Color

# Initialize timer and completion counters
tstart=$(date +%s)
nRunErr=0
nSerRun=0

# Set execution parameters
ITERS=200
INTER=10
CORES=1
COREMAX=$(nproc)
if [[ $CORES -gt $COREMAX ]]
then
	CORES=$COREMAX
fi

problems=$(pwd)

# Define the directories to execute, in order. Formatting matters.
exdirs=("periodic/" \
"no-flux/" \
"T-shape/")

exlabels=("a" \
"b" \
"c")

echo -n "Building problems in serial and parallel"

while [[ $# -gt 0 ]]
do
	key="$1"
	case $key in
		--clean)
			echo -n ", cleaning up after"
		CLEAN=true
		;;
		--short)
			ITERS=$(($ITERS/10))
			INTER=$ITERS
		;;
		--long)
			ITERS=$((10*$ITERS))
			INTER=$(($ITERS/10))
		;;
		--extra)
			ITERS=$((100*$ITERS))
			INTER=$(($ITERS/100))
		;;
		--np)
			shift
			CORES=$1
		;;
		*)
			echo "WARNING: Unknown option ${key}."
			echo
    ;;
	esac
	shift # pop first entry from command-line argument list, reduce $# by 1
done

echo
echo "--------------------------------------------------------------------------"

rm -rf ./*/meta.yml ./*/error.log
codeversion=$(python -c 'from fipy import __version__ as fv; print fv')
repoversion=$(git rev-parse --verify HEAD)
cpufreq=$(lscpu | grep "max MHz" | awk '{print $NF}')
if [[ $cpufreq == "" ]]
then
	cpufreq=$(grep -m1 MHz /proc/cpuinfo | awk '{print $NF}')
fi
sumspace=32

n=${#exdirs[@]}
for (( i=0; i<$n; i++ ))
do
	exstart=$(date +%s)
	j=$(($i+1))
	printf "%s %-${sumspace}s" ${exlabels[$i]} ${exdirs[$i]}
	cd $problems/${exdirs[$i]}

	# Write simulation particulars. Should work on any Debian-flavored GNU/Linux OS.
	echo "---" >>meta.yml
	echo "benchmark:" >>meta.yml
	echo "  id: 1${exlabels[$i]}" >>meta.yml
	echo "" >>meta.yml
	echo "metadata:" >>meta.yml
	echo "  # Describe the runtime environment" >>meta.yml
	echo "  summary: Serial workstation benchmark with FiPy, ${exdirs[$i]/\//} domain" >>meta.yml
	echo "  author: Trevor Keller" >>meta.yml
	echo "  email: trevor.keller@nist.gov" >>meta.yml
	echo "  date: $(date -R)" >>meta.yml
	echo "  hardware:" >>meta.yml
	echo "    # Required hardware details" >>meta.yml
	echo "    architecture: $(uname -m)" >>meta.yml
	echo "    cores: ${CORES}" >>meta.yml
	echo "    # Optional hardware details" >>meta.yml
	echo "    details:" >>meta.yml
	echo "      - name: clock" >>meta.yml
	echo "        value: ${cpufreq}" >>meta.yml
	echo "        units: MHz" >>meta.yml
	echo "  software:" >>meta.yml
	echo "    name: fipy" >>meta.yml
	echo "    url: https://github.com/usnistgov/fipy" >>meta.yml
	echo "    version: $(echo ${codeversion} | head -c 3)" >>meta.yml
	echo "    repo:" >>meta.yml
	echo "      url: https://github.com/mesoscale/mmsp/tree/develop" >>meta.yml
	echo "      version: ${codeversion}" >>meta.yml
	echo "      branch: develop" >>meta.yml
	echo "  implementation:" >>meta.yml
	echo "    end_condition: time limit" >>meta.yml
	echo "    repo:" >>meta.yml
	echo "      url: https://github.com/usnistgov/FiPy-spinodal-decomposition-benchmark/tree/master/${exdirs[$i]}" >>meta.yml
	echo "      version: ${repoversion}" >>meta.yml
	echo "      branch: master" >>meta.yml
	echo "    details:" >>meta.yml
	echo "      - name: mesh" >>meta.yml
	echo "        value: uniform rectilinear" >>meta.yml
	echo "      - name: numerical_method" >>meta.yml
	echo "        value: finite volume" >>meta.yml
	echo "" >>meta.yml

	# Run the example
	rm -f test.*.dat
	(/usr/bin/time -f "  - name: run_time\n    values: {'time': %e, 'unit': seconds}\n  - name: memory_usage\n    values: {'value': %M, 'unit': KB}" bash -c \
	"python cahn-hilliard.py $ITERS $INTER 1>>meta.yml 2>>error.log") &>>meta.yml
	# Return codes are not reliable. Save errors to disk for postmortem.

	if [[ -f error.log ]] && [[ $(wc -w error.log) > 1 ]]
	then
		# Execution failed.
		for i in `seq 1 ${sumspace}`;
		do
			echo -n " "
		done
		echo -e "${RED} --FAILED--${WHT}"
		((nRunErr++))
		if [[ -f error.log ]]
		then
			echo "      error.log has the details (head follows)"
			head error.log | sed -e 's/^/      /'
		fi
	else
		# Execution succeeded.
		((nSerRun++))
		rm -f error.log
		exfin=$(date +%s)
		exlapse=$(echo "$exfin-$exstart" | bc -l)
		printf "${GRN}%${sumspace}d seconds${WHT}\n" $exlapse
	fi

	# Clean up data
	if [[ $CLEAN ]]
	then
		rm -rf data
	fi
done

cd ${problems}

tfinish=$(date +%s)
elapsed=$(echo "$tfinish-$tstart" | bc -l)
echo "--------------------------------------------------------------------------"
printf "Elapsed time: %52d seconds\n" $elapsed
echo
printf "\n%2d problems executed successfully" $nSerRun
if [[ $nRunErr > 0 ]]
then
	printf ", %2d failed" $nRunErr
fi
echo
cd ${problems}

AllERR=$(echo "$nRunErr" | bc -l)
if [[ $AllERR > 0 ]]
then
	echo
	echo "Build error(s) detected: ${AllERR} tests failed."
	exit 1
fi
