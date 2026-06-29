FROM squidfunk/mkdocs-material:latest

# Install plugins.
RUN apk add gcc git python3-dev musl-dev linux-headers
COPY docs/requirements.txt /tmp/
RUN pip install --disable-pip-version-check --no-cache-dir -r /tmp/requirements.txt
