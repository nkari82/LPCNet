# Place in 16k-LP7 from TSPSpeech.iso and run to concatenate wave files
# into one headerless training file
if [ $# -ne 2 ]; then
	echo "Usage: $0 ./header_removal [inputs path] [outputs path]"
	exit -1
else
	echo "ok" 
fi

for i in $1/*.s16 
do
./dump.exe -test:taco $i $2/${i##*/}.f32
done
