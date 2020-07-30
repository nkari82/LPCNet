# Place in 16k-LP7 from TSPSpeech.iso and run to concatenate wave files
# into one headerless training file
if [ $# -ne 2 ]; then
	echo "Usage: $0 ./header_removal [inputs path] [outputs path]"
	exit -1
else
	echo "ok" 
fi

for i in $1/*.wav
do
sox $i -r 16000 -c 1 -t sw - > $2/${i##*/}.s16 
done
