#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
BIN=${DIR}/LinuxNoEditor

# Register binary location
DEEPDRIVE_LOCATION_REG_FILE="${HOME}/.deepdrive/location"
mkdir -p "${DEEPDRIVE_LOCATION_REG_FILE%/*}"
echo ${DIR} > ${DEEPDRIVE_LOCATION_REG_FILE}
echo "Set Deepdrive binary location in ${DEEPDRIVE_LOCATION_REG_FILE} to:
    ${DIR}
"

# Change Unreal's default shipping binary name
mv ${BIN}/DeepDrive/Binaries/Linux/DeepDrive-Linux-Shipping ${BIN}/DeepDrive/Binaries/Linux/DeepDrive &> /dev/null

# Execute!
chmod +x ${BIN}/DeepDrive/Binaries/Linux/DeepDrive
${BIN}/DeepDrive/Binaries/Linux/DeepDrive  $@