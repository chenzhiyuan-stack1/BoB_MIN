FROM challenge-env

USER root
RUN apt-get update && apt-get install -y iproute2