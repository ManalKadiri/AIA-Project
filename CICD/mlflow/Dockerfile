FROM python:3.9-slim

WORKDIR /home/app

# Install dependencies and git
RUN apt-get update && apt-get install -y curl unzip git && \
    curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && ./aws/install && rm -rf awscliv2.zip aws && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# Upgrade pip to the latest version
RUN pip install --upgrade pip

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /home/app/artifacts /home/app/tmp /home/app/mlruns

COPY train.py /home/app/train.py

CMD ["python", "train.py"]
