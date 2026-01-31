FROM python:3.10

ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/pip \
    python -m pip install --upgrade pip && \
    python -m pip install \
    "aiohttp==3.8.4" \
    "aiodocker==0.21.0" \
    agentrl-worker

COPY ./src/server/tasks/os_interaction /app/src/server/tasks/os_interaction
COPY ./configs/tasks/os.yaml /app/configs/tasks/os.yaml
COPY ./data/os_interaction /app/data/os_interaction

COPY --chmod=0755 ./extra/worker-entrypoint.sh /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh", "-c", "configs/tasks/os.yaml"]
