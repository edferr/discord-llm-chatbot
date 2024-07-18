FROM python:3.11

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV OPENAI_API_KEY=XYZ
ENV OPENAI_API_BASE=http://172.17.0.2:1337/v1

# Update package lists and install prerequisites
RUN apt-get update && \
    apt-get install -y \
        software-properties-common \
        wget \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libncursesw5-dev \
        libreadline-dev \
        libsqlite3-dev \
        libgdbm-dev \
        libbz2-dev \
        libexpat1-dev \
        liblzma-dev \
        tk-dev \
        libffi-dev \
        uuid-dev \
    && apt-get clean


# Copy your application code to the container
COPY . /app

# Set the working directory
WORKDIR /app

# Downgrade pip and setuptools --upgrade pip==24.1.2 setuptools==68.2.2
RUN python -m pip install --upgrade pip wheel

# Install Python dependencies from requirements.txt
RUN python -m pip install --use-pep517 --no-cache-dir -r requirements.txt

# Make port 5000 available to the world outside this container (if needed)
EXPOSE 5000

# Run your application
CMD ["python", "llmcord.py"]
