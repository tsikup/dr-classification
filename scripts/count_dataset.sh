data_dir=${1:-"/media/hdd/tsiknakisn/SeeFar/Kaggle/EyePACS/all_data_preprocessed"}

tr0="$(ls $data_dir/train/0 | wc -l)" || tr0=-1
tr1="$(ls $data_dir/train/1 | wc -l)" || tr1=-1
tr2="$(ls $data_dir/train/2 | wc -l)" || tr2=-1
tr3="$(ls $data_dir/train/3 | wc -l)" || tr3=-1
tr4="$(ls $data_dir/train/4 | wc -l)" || tr4=-1


ts0="$(ls $data_dir/test/0 | wc -l)" || ts0=-1
ts1="$(ls $data_dir/test/1 | wc -l)" || ts1=-1
ts2="$(ls $data_dir/test/2 | wc -l)" || ts2=-1
ts3="$(ls $data_dir/test/3 | wc -l)" || ts3=-1
ts4="$(ls $data_dir/test/4 | wc -l)" || ts4=-1

echo Data directory: $data_dir

echo Train/0: $tr0 
echo Train/1: $tr1 
echo Train/2: $tr2 
echo Train/3: $tr3 
echo Train/4: $tr4

echo Test/0: $ts0 
echo Test/1: $ts1 
echo Test/2: $ts2 
echo Test/3: $ts3 
echo Test/4: $ts4 

echo Sum: $(($tr0 + $tr1 + $tr2 + $tr3 + $tr4 + $ts0 + $ts1 + $ts2 + $ts3 + $ts4))
