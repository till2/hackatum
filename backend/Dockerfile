FROM mambaorg/micromamba:2.0.3
USER $MAMBA_USER
WORKDIR /backend
COPY --chown=$MAMBA_USER:$MAMBA_USER ./env.yml .
RUN micromamba install -y -n base -f ./env.yml && \
    micromamba clean --all --yes
COPY . .
EXPOSE 5000
ARG MAMBA_DOCKERFILE_ACTIVATE=1
CMD ["fastapi" , "run", "--host"]
