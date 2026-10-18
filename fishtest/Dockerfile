# Local fishtest for Maskfish in a container.
#
# Built from the root of the repository (the referee uses two helpers of
# trainer/):
#
#     docker build -f fishtest/Dockerfile -t maskfish-fishtest .
#     docker run -d -p 6543:6543 -v fishtest-data:/data maskfish-fishtest
#
# The engines are compiled inside the container when a test starts (git, make
# and g++ are included), with ARCH=native by default: the builds match the CPU
# of the machine running the container. Everything is kept in /data (database,
# git cache, builds, networks, games). Options of fishtest.py can be appended,
# e.g. `docker run ... maskfish-fishtest --concurrency 15 --spsa`; most of them
# can also be changed later on the Settings page.

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git make g++ curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY fishtest/requirements.txt fishtest/requirements.txt
RUN pip install --no-cache-dir -r fishtest/requirements.txt

COPY trainer/jieqi_features.py trainer/jieqi_uci.py trainer/
COPY fishtest/ fishtest/

ENV PYTHONUNBUFFERED=1
VOLUME /data
EXPOSE 6543

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:6543/tests', timeout=4)" || exit 1

ENTRYPOINT ["python", "fishtest/fishtest.py", "--host", "0.0.0.0", "--port", "6543", "--data", "/data"]
