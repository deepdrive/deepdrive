ARG BASE_IMAGE=deepdriveio/deepdrive:3.0
FROM ${BASE_IMAGE}
ARG weights_dir=/home/${DEEPDRIVE_USER}/deepdrive_weights
ARG weights_name="domain_randomization_2019-03-04__02-11-53PM"
RUN mkdir $weights_dir
WORKDIR $weights_dir
RUN wget "https://s3-us-west-1.amazonaws.com/deepdrive/weights/$weights_name.zip"
RUN unzip "$weights_name.zip" -d $weights_dir/$weights_name
RUN ln -sf $weights_dir/$weights_name $weights_dir/domain_randomization
WORKDIR $DEEPDRIVE_SRC_DIR
COPY run.sh run.sh
CMD ./run.sh


