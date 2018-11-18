echo ""
echo "$0"
date

. ./cmd.sh  # Needed for local or cluster-based processing
. ./path.sh # Needed for KALDI_ROOT, REC_ROOT and WAV_ROOT
. ./local/check_prerequisites.sh || exit 1

stage=0

# Define number of parallel jobs
nj=32

feat=av2

eval_list="devel test" # sets used for decoding

# Customize audio data, e.g., setup mixed training.
# You can provide a list of subfolders for each set (train, devel, test).
subfolders_train="isolated"
#subfolders_train="isolated reverberated"

subfolders_test="isolated"
subfolders_devel="isolated"

# Parse external parameters into bash variables
. parse_options.sh 

featdir=$REC_ROOT/$feat

# Setup feature directories
fbank="$REC_ROOT/fbank"
av2="$REC_ROOT/av2"
video="$REC_ROOT/video"
# Setup model and decoding directory
exp="$REC_ROOT/exp"
mkdir -p $exp

# Setup other relevant directories
data="$REC_ROOT/data"
lang="$data/lang"
dict="$data/local/dict"
langtmp="$data/local/lang"
mkdir -p $langtmp
steps="steps"
utils="utils"
boost_silence=1.0

# print setup
echo "Features: ${feat}"
echo "Training data: ${subfolders_train}"
echo "Test data: ${eval_list}"

# Data preparation
if [ $stage -le 0  ]; then
	echo ""
	echo "Stage ${stage}: Preparing data"

	# create a copy of the utilized audio data using symbolic links
	WAV_ROOT_TMP=$REC_ROOT/wav
	rm -rf $WAV_ROOT_TMP/*

	echo "Creating symbolic links to audio directories (train, test, devel) at $WAV_ROOT_TMP";
	mkdir -p $WAV_ROOT_TMP/train $WAV_ROOT_TMP/devel $WAV_ROOT_TMP/test

	for sf in $subfolders_train; do
		ln -snfv $WAV_ROOT/train/$sf $WAV_ROOT_TMP/train/$sf || exit 1
	done
	for sf in $subfolders_test; do
		ln -snfv $WAV_ROOT/test/$sf $WAV_ROOT_TMP/test/$sf || exit 1
	done
	for sf in $subfolders_devel; do
		ln -snfv $WAV_ROOT/devel/$sf $WAV_ROOT_TMP/devel/$sf || exit 1
	done
	
	#WAV_ROOT=$WAV_ROOT_TMP
	rm -rf $data/*
	local/chime1_prepare_data.sh --WAV_ROOT $WAV_ROOT_TMP || exit 1
fi


# Language model preparation
if [ $stage -le 1  ]; then
	echo ""
	echo "Stage ${stage}: Preparing language"
	local/chime1_prepare_dict.sh || exit 1
	$utils/prepare_lang.sh --num-sil-states 5 \
		--num-nonsil-states 3 \
		--position-dependent-phones false \
		--share-silence-phones true \
		$dict "A" $langtmp $lang || exit 1
	local/chime1_prepare_grammar.sh || exit 1
fi

# Feature extraction
if [ $stage -le 2  ]; then
	echo ""
	echo "Stage ${stage}: Extracting features"
	
	# TODO: Check how this can be cleaned
	rm -rf $featdir/*
	mkdir -p $featdir

	rm -rf $REC_ROOT/$feat/*
	mkdir -p $REC_ROOT/$feat

	rm -rf $REC_ROOT/tmp/*
	mkdir -p $REC_ROOT/tmp

	fe_list="train test devel" # sets used for feature extraction
	
	for x in $fe_list; do
		# extract regular features
	      mkdir -p $fbank
		  data2=$data/fbank
		  rm -rf $data2/$x/*
		  mkdir -p $data2/$x
		  cp -R $data/$x/* $data2/$x


		  $steps/make_fbank.sh --nj $nj --cmd "$train_cmd" --fbank_config conf/fbank.conf $data2/$x $exp/make_fbank/$x $fbank || exit 1
		  # Compute CMVN stats
		  $steps/compute_cmvn_stats.sh $data2/$x $exp/make_fbank/$x $fbank || exit 1

		  mkdir -p $video
		  data2=$data/video
		  rm -rf $data2/$x/*
		  mkdir -p $data2/$x
		  cp -R $data/$x/* $data2/$x

		  echo "Running make_video.sh"
		  local/make_video.sh --nj $nj \
			  --cmd "$train_cmd" \
			  --audioRoot $REC_ROOT/wav \
			  --videoRoot $VIDEO_ROOT \
			  $data2/$x \
			  $exp/make_video/$x \
			  $video || exit 1

		  # Compute CMVN stats
		  $steps/compute_cmvn_stats.sh $data2/$x $exp/make_video/$x $video || exit 1

		  data2=$data/$feat
		  mkdir -p $data2

		  # Append audio/video features
		  $steps/append_feats.sh --nj $nj --cmd "$train_cmd" \
			  $data/fbank/$x $data/video/$x \
			  $data2/$x $exp/make_${feat}/$x $featdir || exit 1
	  
	  $steps/compute_cmvn_stats.sh $data2/$x $exp/make_${feat}/$x $featdir || exit 1
  done
fi

data=$data/$feat
exp=$exp/$feat
mkdir -p $exp

# GMM Training
if [ $stage -le 3 ]; then
	echo ""
	echo "Stage 3: Starting training"
	rm $data/train/split*

	$steps/train_mono.sh --nj $nj --cmd "$train_cmd" \
		--boost_silence $boost_silence \
		$data/train $lang $exp/mono0a || exit 1;

	$steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		--boost_silence $boost_silence \
		$data/train $lang $exp/mono0a $exp/mono0a_ali || exit 1;

	$steps/train_deltas.sh --cmd "$train_cmd" \
		--boost_silence $boost_silence \
		2000 10000 $data/train $lang $exp/mono0a_ali $exp/tri1 || exit 1;

	$steps/align_si.sh --nj $nj --cmd "$train_cmd" \
		$data/train $lang $exp/tri1 $exp/tri1_ali || exit 1;

	$steps/train_lda_mllt.sh --cmd "$train_cmd" \
		--splice-opts "--left-context=3 --right-context=3" \
		2500 15000 $data/train $lang $exp/tri1_ali $exp/tri2b || exit 1;

	$utils/mkgraph.sh $lang $exp/tri2b $exp/tri2b/graph || exit 1;
fi

if ! cuda-compiled; then
	cat <<EOF && exit 1
	this script is intended to be used with GPUS.
EOF
fi

train_set=train
test_sets="devel test"
gmm=tri2b
audio_feat=fbank

affix=1a
common_egs_dir=

# LSTM/chain options
train_stage=-10
xent_regularize=0.1

# training chunk-options
chunk_width=140,100,160
chunk_left_context=0
chunk_right_context=0

# training options
srand=0
remove_egs=false
data=$REC_ROOT/data
if [ $stage -le 4 ]; then
	echo "preparing directory for speed-perturbed data"
	./utils/data/perturb_data_dir_speed_3way.sh $data/${audio_feat}/${train_set} $data/${audio_feat}/${train_set}_sp_vol
	./utils/data/perturb_data_dir_speed_3way.sh $data/${audio_feat}/${train_set} $data/${audio_feat}/${train_set}_sp
	./utils/data/perturb_data_dir_volume.sh $data/${audio_feat}/${train_set}_sp_vol
	./utils/fix_data_dir.sh $data/${audio_feat}/${train_set}_sp
fi

if [ $stage -le 5 ]; then
	if [ ! -d ./video/train/isolated/sp0.9-id1 ]; then
		cat ./dup_list | parallel --eta 
	fi

	echo "make sp fbank and append audio & video feature"
	# Compute fbank feature
	$steps/make_fbank.sh --nj $nj --cmd "$train_cmd" --fbank_config conf/fbank.conf $data/$audio_feat/${train_set}_sp $exp/make_fbank_sp $REC_ROOT/fbank_sp
	$steps/make_fbank.sh --nj $nj --cmd "$train_cmd" --fbank_config conf/fbank.conf $data/$audio_feat/${train_set}_sp_vol $exp/make_fbank_sp_vol $REC_ROOT/fbank_sp_vol

	# Compute CMVN stats
	$steps/compute_cmvn_stats.sh $data/$audio_feat/${train_set}_sp $exp/make_fbank_sp $REC_ROOT/fbank_sp
	$steps/compute_cmvn_stats.sh $data/$audio_feat/${train_set}_sp_vol $exp/make_fbank_sp_vol $REC_ROOT/fbank_sp_vol
	
	cp -r $data/${audio_feat}/${train_set}_sp $data/video_sp

	# make video feat.scp 
	video_sp=$REC_ROOT/video_sp
	local/make_video_sp.sh --nj $nj \
		--cmd "$train_cmd" \
		--audioRoot $REC_ROOT/wav \
		--videoRoot $VIDEO_ROOT \
		$data/video_sp \
		$exp/make_video_sp/$x \
		$video_sp

	$steps/compute_cmvn_stats.sh  $data/video_sp $exp/make_video_sp/$x $video_sp

	mkdir -p $data/append_sp
	mkdir -p $data/append_sp_vol

	# Append audio/video features
	$steps/append_feats.sh --nj $nj --cmd "$train_cmd" \
		$data/$audio_feat/${train_set}_sp $data/video_sp \
		$data/append_sp $exp/make_append_sp $REC_ROOT/append_sp
    
	$steps/append_feats.sh --nj $nj --cmd "$train_cmd" \
		$data/$audio_feat/${train_set}_sp_vol $data/video_sp \
		$data/append_sp_vol $exp/make_append_sp_vol $REC_ROOT/append_sp_vol

	$steps/compute_cmvn_stats.sh $data/append_sp $exp/make_append_sp $REC_ROOT/append_sp
	$steps/compute_cmvn_stats.sh $data/append_sp_vol $exp/make_append_sp_vol $REC_ROOT/append_sp_vol

fi

lat_dir=exp/${feat}/chain${nnet3_affix}/${gmm}_${train_set}_sp_lat
gmm_dir=exp/${feat}/${gmm}
ali_dir=exp/${feat}/${gmm}_ali_${train_set}_sp
dir=exp/${feat}/chain${nnet3_affix}/tdnn${affix}_sp
train_data_dir=$REC_ROOT/data/append_sp_vol
lores_train_data_dir=$REC_ROOT/data/append_sp

# note: you don't necessarily have to change the treedir name
# each time you do a new experiment-- only if you change the
# configuration in a way that affects the tree.
tree_dir=exp/${feat}/chain${nnet3_affix}/tree_a_sp
# the 'lang' directory is created by this script.
# If you create such a directory with a non-standard topology
# you should probably name it differently.
lang=data/lang_chain

if [ $stage -le 6  ]; then
	echo "aligning with the perturbed low-resolution data"
	steps/align_fmllr.sh --nj $nj --cmd "$train_cmd" \
		 $lores_train_data_dir data/lang $gmm_dir $ali_dir

	for f in $train_data_dir/feats.scp \
		$lores_train_data_dir/feats.scp $gmm_dir/final.mdl \
		$ali_dir/ali.1.gz $gmm_dir/final.mdl; do
		[ ! -f $f  ] && echo "$0: expected file $f to exist" && exit 1
	done
fi

if [ $stage -le 7  ]; then
echo "$0: creating lang directory $lang with chain-type topology"
# Create a version of the lang/ directory that has one state per phone in the
# topo file. [note, it really has two states.. the first one is only repeated
# once, the second one has zero or more repeats.]
if [ -d $lang  ]; then
	if [ $lang/L.fst -nt data/lang/L.fst  ]; then
		echo "$0: $lang already exists, not overwriting it; continuing"
	else
		echo "$0: $lang already exists and seems to be older than data/lang..."
		echo " ... not sure what to do.  Exiting."
		exit 1;
	fi
else
	cp -r data/lang $lang
	silphonelist=$(cat $lang/phones/silence.csl) || exit 1;
	nonsilphonelist=$(cat $lang/phones/nonsilence.csl) || exit 1;
	# Use our special topology... note that later on may have to tune this
	# topology.
	steps/nnet3/chain/gen_topo.py $nonsilphonelist $silphonelist >$lang/topo
    fi
fi

if [ $stage -le 8  ]; then
	# Get the alignments as lattices (gives the chain training more freedom).
	# use the same num-jobs as the alignments
	steps/align_fmllr_lats.sh --nj $nj --cmd "$train_cmd" ${lores_train_data_dir} \
		data/lang $gmm_dir $lat_dir
	rm $lat_dir/fsts.*.gz # save space
fi

if [ $stage -le 9  ]; then
	# Build a tree using our new topology.  We know we have alignments for the
	# speed-perturbed data (local/nnet3/run_ivector_common.sh made them), so use
	# those.  The num-leaves is always somewhat less than the num-leaves from
	# the GMM baseline.
	rm -rf $tree_dir
	if [ -f $tree_dir/final.mdl  ]; then
		echo "$0: $tree_dir/final.mdl already exists, refusing to overwrite it."
		exit 1;
	fi
	steps/nnet3/chain/build_tree.sh \
    	--frame-subsampling-factor 3 \
		--context-opts "--context-width=2 --central-position=1" \
		--cmd "$train_cmd" 3500 ${lores_train_data_dir} \
		$lang $ali_dir $tree_dir
fi

if [ $stage -le 10  ]; then
	mkdir -p $dir
	echo "$0: creating neural net configs using the xconfig parser";

	num_targets=$(tree-info $tree_dir/tree |grep num-pdfs|awk '{print $2}')
	learning_rate_factor=$(echo "print 0.5/$xent_regularize" | python)

	mkdir -p $dir/configs
	cat <<EOF > $dir/configs/network.xconfig
	input dim=103 name=input

	# please note that it is important to have input layer with the name=input
	# as the layer immediately preceding the fixed-affine-layer to enable
	# the use of short notation for the descriptor
	fixed-affine-layer name=lda input=Append(-2,-1,0,1,2) affine-transform-file=$dir/configs/lda.mat

	# the first splicing is moved before the lda layer, so no splicing here
	relu-batchnorm-layer name=tdnn1 dim=750
	relu-batchnorm-layer name=tdnn2 dim=750 input=Append(-1,0,1)
	relu-batchnorm-layer name=tdnn3 dim=750
	relu-batchnorm-layer name=tdnn4 dim=750 input=Append(-1,0,1)
	relu-batchnorm-layer name=tdnn5 dim=750
	relu-batchnorm-layer name=tdnn6 dim=750 input=Append(-3,0,3)
	relu-batchnorm-layer name=tdnn7 dim=750 input=Append(-3,0,3)
	relu-batchnorm-layer name=tdnn8 dim=750 input=Append(-6,-3,0)

	## adding the layers for chain branch
	relu-batchnorm-layer name=prefinal-chain dim=750 target-rms=0.5
	output-layer name=output include-log-softmax=false dim=$num_targets max-change=1.5

	# adding the layers for xent branch
	# This block prints the configs for a separate output that will be
	# trained with a cross-entropy objective in the 'chain' models... this
	# has the effect of regularizing the hidden parts of the model.  we use
	# 0.5 / args.xent_regularize as the learning rate factor- the factor of
	# 0.5 / args.xent_regularize is suitable as it means the xent
	# final-layer learns at a rate independent of the regularization
	# constant; and the 0.5 was tuned so as to make the relative progress
	# similar in the xent and regular final layers.
	relu-batchnorm-layer name=prefinal-xent input=tdnn8 dim=750 target-rms=0.5
	output-layer name=output-xent dim=$num_targets learning-rate-factor=$learning_rate_factor max-change=1.5
EOF
	steps/nnet3/xconfig_to_configs.py --xconfig-file $dir/configs/network.xconfig --config-dir $dir/configs/
fi


if [ $stage -le 11  ]; then
	if [[ $(hostname -f) == *.clsp.jhu.edu  ]] && [ ! -d $dir/egs/storage  ]; then
		utils/create_split_dir.pl \
			/export/b0{3,4,5,6}/$USER/kaldi-data/egs/chime4-$(date +'%m_%d_%H_%M')/s5/$dir/egs/storage $dir/egs/storage
	fi

	steps/nnet3/chain/train.py --stage=$train_stage \
		--cmd="$decode_cmd" \
		--feat.cmvn-opts="--norm-means=false --norm-vars=false" \
		--chain.xent-regularize $xent_regularize \
		--chain.leaky-hmm-coefficient=0.1 \
		--chain.l2-regularize=0.00005 \
		--chain.apply-deriv-weights=false \
		--chain.lm-opts="--num-extra-lm-states=2000" \
		--trainer.srand=$srand \
		--trainer.max-param-change=2.0 \
		--trainer.num-epochs=6 \
		--trainer.frames-per-iter=3000000 \
		--trainer.optimization.num-jobs-initial=1 \
		--trainer.optimization.num-jobs-final=1 \
		--trainer.optimization.initial-effective-lrate=0.003 \
		--trainer.optimization.final-effective-lrate=0.0003 \
		--trainer.optimization.shrink-value=1.0 \
		--trainer.optimization.proportional-shrink=60.0 \
		--trainer.num-chunk-per-minibatch=128,64 \
		--trainer.optimization.momentum=0.0 \
		--egs.chunk-width=$chunk_width \
		--egs.chunk-left-context=0 \
		--egs.chunk-right-context=0 \
		--egs.chunk-left-context-initial=0 \
		--egs.chunk-right-context-final=0 \
		--egs.dir="$common_egs_dir" \
		--egs.opts="--frames-overlap-per-eg 0" \
		--cleanup.remove-egs=$remove_egs \
		--use-gpu=true \
		--reporting.email="$reporting_email" \
		--feat-dir=$train_data_dir \
		--tree-dir=$tree_dir \
		--lat-dir=$lat_dir \
		--dir=$dir  || exit 1;
fi


if [ $stage -le 12  ]; then
	# The reason we are using data/lang here, instead of $lang, is just to
	# emphasize that it's not actually important to give mkgraph.sh the
	# lang directory with the matched topology (since it gets the
	# topology file from the model).  So you could give it a different
	# lang directory, one that contained a wordlist and LM of your choice,
	# as long as phones.txt was compatible.

	#utils/lang/check_phones_compatible.sh \
		#  data/lang_test_tgpr_5k/phones.txt $lang/phones.txt
	#utils/mkgraph.sh \
		#  --self-loop-scale 1.0 data/lang_test_tgpr_5k \
		#  $tree_dir $tree_dir/graph_tgpr_5k || exit 1;
	utils/mkgraph.sh \
		--self-loop-scale 1.0 data/lang \
		$tree_dir $tree_dir/graph || exit 1;
fi

if [ $stage -le 13  ]; then
	frames_per_chunk=$(echo $chunk_width | cut -d, -f1)
	rm $dir/.error 2>/dev/null || true

	for data in $test_sets; do
		(
		data_affix=$(echo $data | sed s/test_//)
		nspk=$(wc -l <data/${data}/spk2utt)
		#for lmtype in basic; do
		steps/nnet3/decode.sh \
			--acwt 1.0 --post-decode-acwt 10.0 \
			--extra-left-context 0 --extra-right-context 0 \
			--extra-left-context-initial 0 \
			--extra-right-context-final 0 \
			--frames-per-chunk $frames_per_chunk \
			--nj $nspk --cmd "$decode_cmd"  --num-threads 4 \
			$tree_dir/graph data/${feat}/${data} ${dir}/decode_${data_affix} || exit 1
			#$tree_dir/graph data/${data}_hires ${dir}/decode_${data_affix} || exit 1
		#$tree_dir/graph_${lmtype} data/${data}_hires ${dir}/decode_${lmtype}_${data_affix} || exit 1
		#done
		) || touch $dir/.error &
	done
	wait
	[ -f $dir/.error  ] && echo "$0: there was a problem while decoding" && exit 1
fi

exit 0;



