# Use the official Python image from the Docker Hub
FROM python:3.10-slim

# Set the working directory in the container
WORKDIR /app

# Copy the requirements.txt file into the container at /app
COPY requirements.txt requirements.txt

# Install the required packages
RUN pip install --no-cache-dir --default-timeout=100 -r requirements.txt

# Copy the rest of the application code into the container at /app
COPY . .

# Expose port 5000 to the outside world
EXPOSE 5000

# Run the application with Waitress
CMD ["python" , "flaskapp/app.py"]

