#!/bin/bash

# Use paramert:
#	1 - path to detector
#   2 - folder to save result (near config.yml)
# 	3 - use cpu or gpu (cpu or gpu)
#	4 - use mpi (0 or 1), if use cpu
#	5 - n (if mpi use)

if [[ -z "$1" ||  -z "$2" || -z "$3" ]] ; then 
	echo $'1 - path to detector \n2 - folder to save result\n3 - use cpu or gpu (0 or 1)\n4 - use mpi (0 or 1), if use cpu\n5 - n (if mpi use)';
	exit ;
fi

path_to_datector=$1
folder_to_save=$2
use_gpu=$3
count__proc="1"
if [ "$use_gpu" == "cpu" ]; then 
	use_mpi=$4
	use_gpu="all"
	if [ $use_mpi -eq 1 ]; then
		count__proc=$5
	fi
fi

config_file=config.yml

echo "PATH 	to detector" $path_to_datector 
echo "Save folder" $folder_to_save
echo "Use cpu or gpu" $use_gpu
echo "config file" $config_file

i="0"
while [ $i -lt 10 ]; do
	i=$[$i+1]
	if [ $i -eq 10 ]; then 
		name_use=$folder_to_save"/fold-"$i"-out.txt"
		name_fold=$folder_to_save"/fddb/FDDB-fold-"$i".txt"
	else 
		name_use=$folder_to_save"/fold-0"$i"-out.txt"
		name_fold=$folder_to_save"/fddb/FDDB-fold-0"$i".txt"
	fi
	echo "Use fold out number" $i "with name " $name_use
	sed -i '2 c input_file_name: '"$name_fold" $config_file
	sed -i '$ c output_file_name: '"$name_use"  $config_file

	#run detector
	if [ $use_mpi -eq 1 ];then
		#echo $count__proc
		srun -t "3-00:00:00" -o $i".txt" -p $use_gpu -N $count__proc $path_to_datector"/caffe_detector"  $folder_to_save  &
	else
		srun -t "3-00:00:00" -o $i".txt" -p $use_gpu -N $count__proc $path_to_datector"/caffe_detector"  $folder_to_save  &
	fi

done