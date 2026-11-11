# Local fishtest for Maskfish in a container.
#
# Built from the root of the repository (the games are played by tools/match.py
# with the referee of tools/native and tools/jieqi):
#
#     docker build -f fishtest/Dockerfile -t maskfish-fishtest .
#     docker run -d -p 6543:6543 -v fishtest-data:/data maskfish-fishtest
#
# The engines are compiled inside the container when a test starts (git, cmake
# and g++ are included), with -march=native by default: the builds match the
# CPU of the machine running the container. The referee's library is built into
# the image (/app/build/lib). Everything else is kept in /data (database, git
# cache, builds, networks, games). Options of fishtest.py can be appended, e.g.
# `docker run ... maskfish-fishtest --concurrency 15 --spsa`; most of them can also
# be changed later on the Settings page.

FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends git cmake make g++ curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY fishtest/requirements.txt fishtest/requirements.txt
RUN pip install --no-cache-dir -r fishtest/requirements.txt

COPY tools/native/ tools/native/
RUN mkdir -p build/lib \
    && g++ -std=c++20 -O2 -fPIC -shared -DJIEQI_BUILD_DLL -Itools/native/include \
       tools/native/c_api.cpp tools/native/rules.cpp tools/native/repetition.cpp tools/native/chase.cpp \
       -o build/lib/libjieqi_rules.so
COPY tools/jieqi/ tools/jieqi/
COPY tools/match.py tools/sprt.py tools/
COPY fishtest/ fishtest/

ENV PYTHONUNBUFFERED=1
ENV JIEQI_NATIVE_LIBRARY=/app/build/lib/libjieqi_rules.so
VOLUME /data
EXPOSE 6543

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:6543/tests', timeout=4)" || exit 1

ENTRYPOINT ["python", "fishtest/fishtest.py", "--host", "0.0.0.0", "--port", "6543", "--data", "/data"]
