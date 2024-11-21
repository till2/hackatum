FROM node:18-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY ./frontend .
RUN npx vite build



FROM mambaorg/micromamba:2.0.3 AS backend
USER $MAMBA_USER
WORKDIR /app
COPY --chown=$MAMBA_USER:$MAMBA_USER ./backend/env.yml .
RUN micromamba install -y -n base -f ./env.yml && \
    micromamba clean --all --yes
COPY ./backend .
COPY --from=frontend-builder /app/frontend/dist ./static
EXPOSE 8000
ARG MAMBA_DOCKERFILE_ACTIVATE=1
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

