
#!/bin/bash


SOURCE_DIR="/home/ecasanova/Projects/Papers/ICASSP-2025-21Hz-codec/NeMo-Speech-Codec/audios_demo/T5-TTS/"
TARGET_DIR="/home/ecasanova/Projects/Papers/ICASSP-2025-21Hz-codec/NeMo-Speech-Codec/audios_demo/T5-TTS_22kHz/"



OUT_EXTENSION=".wav"
SEARCH_EXTENSION=".wav"
export SOURCE_DIR
export TARGET_DIR
export OUT_EXTENSION
export SEARCH_EXTENSION
doone() {
    inputFile="$1"
    if [[ "$(basename "${inputFile}")" != ._* ]] ; then # Skip files starting with "._"
        tmpVar="${inputFile}"
        outFile="${tmpVar/$SOURCE_DIR/$TARGET_DIR}"
        outFile="${outFile/$SEARCH_EXTENSION/$OUT_EXTENSION}"
        outFilePath=$(dirname "${outFile}")
        mkdir -p "${outFilePath}"
        # if [ ! -f "$outFile" ]; then # If the mp3 file doesn't exist already
        echo "Input: $inputFile"
        echo "Output: $outFile"
        ffmpeg -y -hide_banner -loglevel error -i "$inputFile" -ac 1 -ar 22050 "$outFile"
    fi
}

dofiles() {
    inputFile="$1"
    if [[ "$(basename "${inputFile}")" != ._* ]] ; then # Skip files starting with "._"
        tmpVar="${inputFile}"
        outFile="${tmpVar/$SOURCE_DIR/$TARGET_DIR}"
        outFilePath=$(dirname "${outFile}")
        mkdir -p "${outFilePath}"
        if [ ! -f "$outFile" ]; then # If the mp3 file doesn't exist already
            echo "Input: $inputFile"    
            echo "Output: $outFile"
            cp "$inputFile" "$outFile"
        fi
    fi
}


export -f doone
export -f dofiles



# Find all flac/wav files in the given SOURCE_DIR and iterate over them:
find "${SOURCE_DIR}" -type f \( -iname "*${SEARCH_EXTENSION}" \) -print0 |
  parallel -0 doone
  
# texts
# find "${SOURCE_DIR}" -type f \( -iname "*.txt" \) -print0 |
#   parallel -0 dofiles
