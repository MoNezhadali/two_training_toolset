# Load the base image
FROM python:3.11-slim

# Set default time zone
ENV TIME_ZONE=UTC
RUN ln -snf /usr/share/zoneinfo/$TIME_ZONE /etc/localtime && \
    echo $TIME_ZONE > /etc/timezone

# Set the working directory
WORKDIR /scripts

# Copy the application code
COPY src/__init__.py src/__init__.py
COPY src/serve src/serve
COPY requirements.txt .
COPY artifacts artifacts

# Install the python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Run the app
CMD ["python", "-m", "src.serve.app"]
