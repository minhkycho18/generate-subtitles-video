FROM ubuntu:latest

RUN apt-get update && apt-get install -y ffmpeg
# Use the official Python image from the Docker Hub
FROM python:3.10-slim


# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip install --no-cache-dir --ignore-installed   --default-timeout=100 -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

RUN echo "export IMAGEIO_FFMPEG_EXE=ffmpeg" >> /etc/bash.bashrc
# Expose port 5000 to the outside world
EXPOSE 8080

# Run the application with Waitress
CMD ["python" , "flaskapp/app.py"]