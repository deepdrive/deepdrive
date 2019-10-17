FROM deepdriveio/deepdrive:3.0
ENV WEIGHTS_DIR=/home/${DEEPDRIVE_USER}/deepdrive_weights
ENV WEIGHTS_NAME="domain_randomization_2019-03-04__02-11-53PM"
# TODO: Move below to python when adding more bots
RUN mkdir ${WEIGHTS_DIR}
WORKDIR ${WEIGHTS_DIR}
RUN wget "https://s3-us-west-1.amazonaws.com/deepdrive/weights/${WEIGHTS_NAME}.zip"
RUN unzip "${WEIGHTS_NAME}.zip" -d ${WEIGHTS_DIR}/${WEIGHTS_NAME}
RUN ln -sf ${WEIGHTS_DIR}/${WEIGHTS_NAME} ${WEIGHTS_DIR}/domain_randomization
WORKDIR $DEEPDRIVE_SRC_DIR
COPY run.sh run.sh
CMD ./run.sh


