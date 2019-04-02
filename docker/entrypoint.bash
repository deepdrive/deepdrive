#!/bin/bash

#echo "Changing ${DEEPDRIVE_DIR} permissions, including any bind-mounts within"
#sudo find ${DEEPDRIVE_DIR} ! -perm 775 -exec chmod 775 {} +
exec "$@"

#--net-path="/home/c2/Deepdrive/tensorflow/2019-03-04__02-11-53PM" --view-mode-period=16