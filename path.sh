# Default working directory for storing data/ exp/ mfcc/ etc
# Can be redefined under each USER case
export REC_ROOT="."

# CHiME-1/CHiME-2 audio root
# export WAV_ROOT="/mnt/database/CHIME1/PCCdata16kHz_mono"
# export WAV_ROOT="/mnt/public/hendrik_AVSR/audio/chime2-grid/mono"
export WAV_ROOT="/home/yong/DB/[DB]_CHIME1/original_ver/PCCdata16kHz"

# KALDI binary root
export KALDI_ROOT=`pwd`/../../..

#export PATH=$PWD/utils/:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$KALDI_ROOT/src/online2bin/:$KALDI_ROOT/src/ivectorbin/:$KALDI_ROOT/src/lmbin/:$PWD:$PATH
#export PATH=$PWD/src/bin:$PWD/src/featbin:$PATH
. $KALDI_ROOT/tools/config/common_path.sh
export LC_ALL=C
