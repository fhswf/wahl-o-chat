FROM python:3.12-slim

# Set the working directory
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY ./requirements.txt /app

# Install any needed packages specified in requirements.txt
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY . /app

# Make Port 7860 available to the world outside this container
EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"

CMD ["python", "gr_app.py"]