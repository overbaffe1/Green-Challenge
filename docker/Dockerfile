# syntax=docker/dockerfile:1.7
# ============================================================================
# Green Challenge 2026 submission image
#
# The contest image already contains the GreenVLAv1.1 checkpoint, the Qwen
# backbone cache, CUDA kernels and the two official servers.  Rebuilding that
# stack from scratch would add ~20 GB of downloads and a new source of drift,
# so this image starts from the organisers' image and replaces the entrypoint.
#
#   docker build -f docker/Dockerfile \
#       --build-arg BASE_IMAGE='aij-green-challenge.cr.cloud.ru/green-vla@sha256:<digest>' \
#       -t green-challenge:local .
#
# Pin BASE_IMAGE by digest for a submission: a moving tag is not reproducible.
# ============================================================================

ARG BASE_IMAGE=aij-green-challenge.cr.cloud.ru/green-vla:latest
FROM ${BASE_IMAGE}

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    GC_STACK=official \
    GC_SPEED=1.3 \
    GC_ROW_DT=0.03333333333333333 \
    GC_TELEMETRY=/tmp/greenchallenge/tracking.jsonl

# Keep the organisers' entrypoint under a different name: it is the exact
# baseline this image is compared against, and having it in the image makes a
# rollback a one-variable change (see docs/RUNBOOK.md).
RUN if [ -f /app/docker/entrypoint.py ]; then \
        cp /app/docker/entrypoint.py /app/docker/entrypoint.official.py; \
    fi

COPY greenchallenge /app/greenchallenge
COPY docker/entrypoint.py /app/docker/entrypoint.py
COPY docs /app/docs

# Fail the build, not the benchmark, if the import chain is broken.
RUN python -c "import numpy, msgpack, websockets; import greenchallenge; \
print('greenchallenge', greenchallenge.__version__)"

EXPOSE 8999

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=3 \
    CMD ["python", "-c", "import os,sys,urllib.request; p=os.environ.get('POLICY_PORT','8999'); sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+p+'/healthz', timeout=4).status==200 else 1)"]

ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD []
