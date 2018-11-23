#!/bin/bash

. ./path.sh


spk=$1
video_dir=$REC_ROOT/video
dur_dir=./utt2dur
db_type=isolated
dir=$video_dir/train/$db_type
sp_list="0.9 1.1"
#spk_list=`ls $dir`
audio_spk_list=`ls $dir | sed 's/id/s/g'`
spk_num=`ls $dir | sed 's/id//g'`

# uniq data preparation
#for spk in $spk_list; do
	utt_list=`ls $dir/$spk | sed 's/\.[^.]*$//'`
	for sp in $sp_list; do
		mkdir -p $dir/sp$sp'-'$spk
		for utt in $utt_list; do
			# extract uniq video feature
			copy-feats ark:$dir/$spk/$utt.ark ark,t:$dir/sp$sp'-'$spk/$utt.ark
			sed -i "s/]//g" $dir/sp$sp'-'$spk/$utt.ark
			uniq $dir/sp$sp'-'$spk/$utt.ark > $dir/sp$sp'-'$spk/$utt.ark.uniq
			rm $dir/sp$sp'-'$spk/$utt.ark
            mv $dir/sp$sp'-'$spk/$utt.ark.uniq $dir/sp$sp'-'$spk/$utt.ark
			nvframe=`cat $dir/sp$sp'-'$spk/$utt.ark | wc -l`
			nvframe=`echo "$nvframe-1" | bc`
			num=`echo $spk | sed 's/id//g'`
			num=`seq -f "%02g" $num $num`
			audio_spk=`echo s$num`
			utt_id=$audio_spk'_'$utt'_'$db_type
			# extract utterance duration
			dur=`grep $utt_id $dur_dir | awk '{print $2}'`
			# estimate num_frame after speed perturbation
			# round((duration * 100) / speed) - 2
			frame_1_0=`echo "$dur*100" | bc`
			frame_sp=`echo "scale=5;$frame_1_0/$sp" | bc`
			frame_sp=`printf %0.f $frame_sp`
			frame_sp=`echo "$frame_sp-2" | bc`
			# duplicate video features by differential digital analyser
			vframe_time=`echo "scale=5;$dur*1000/$nvframe" | bc`
			#frame_time=`echo "scale=5;10/$sp" | bc`
			frame_time=10
			vframe_time=`echo "scale=5;$vframe_time/$sp" | bc`
			video_idx=1
			tmp=$dir/sp$sp'-'$spk/$utt.ark
				# duplicate utt id
			echo sp$sp'-'`cat $tmp | head -1 | tail -1` >> $tmp.tmp
			for idx in $(seq 1 $frame_sp); do
				current_time=`echo "scale=2;$frame_time*$idx" | bc`
				video_time=`echo "scale=2;$vframe_time*$video_idx" | bc`
				# make time value to integer type
				current_time=`echo "$current_time*100"|bc`
				video_time=`echo "$video_time*100"|bc`
				current_time=`printf %.0f $current_time`
				video_time=`printf %.0f $video_time`
				if [ "$current_time" -le "$video_time" ]; then
					tmp_idx=`echo "$video_idx+1" | bc`
					cat $tmp | head -$tmp_idx | tail -1 >> $tmp.tmp
				elif [ "$current_time" -gt "$video_time" ]; then
					video_idx=`echo "$video_idx+1" | bc`
					tmp_idx=`echo "$video_idx+1" | bc`
					cat $tmp | head -$tmp_idx | tail -1 >> $tmp.tmp
				fi
				if [ $idx -eq $frame_sp ]; then
				fi
			done
			rm $tmp
			# add last stiring ( ])
			last_idx=`cat $tmp.tmp | wc -l`
			last=`cat $tmp.tmp | tail -1`
			s=s
			sed -i "$last_idx$s/$last/$last ]/" $tmp.tmp
			#mv $tmp.tmp $tmp
			copy-feats ark,t:$tmp.tmp ark:$tmp
			rm $tmp.tmp
		done
	done
#done

