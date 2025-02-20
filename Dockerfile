# Use an official lightweight Python image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Copy only the requirements.txt first (to cache dependencies)
COPY requirements.txt .

# Install system dependencies for OpenCV and Python dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies before copying the entire app (Docker caching)
RUN pip install --no-cache-dir -r requirements.txt

# Now copy the entire application
COPY . .

# Expose the port the app runs on
EXPOSE 5000

# Run the application
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
