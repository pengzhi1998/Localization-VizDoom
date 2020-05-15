#!/usr/bin/env bash
set -e

rm -rf outputs
mkdir -p outputs
mkdir -p outputs/sources
mkdir -p outputs/images

NUM_MAZES=200

ACC="./acc/acc"
if [ ! -f $ACC ]; then
	echo "File $ACC does not exist, compiling..."
	make -C ./acc
fi

for FILE in content/*.acs; do
	$ACC -i ./acc $FILE "../../VizDoom/$(basename ${FILE%.*}).o"
done

for SIZE in $(seq -s ' ' 7 2 11); do
	PREFIX="../../VizDoom/sources/$SIZE"
	python maze.py $PREFIX -r $SIZE -c $SIZE -n $NUM_MAZES
	python wad.py "${PREFIX}_TRAIN" "../../VizDoom/${SIZE}_TRAIN.wad" -b ../../VizDoom/static_goal.o
	python wad.py "${PREFIX}_TEST" "../../VizDoom/${SIZE}_TEST.wad" -b ../../VizDoom/static_goal.o
done
echo "Success."
