# Use an official Python image as the base
FROM python:3.12

# Set the working directory inside the container
WORKDIR /app

# Copy the application files to the container
COPY . .

# Install required dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 5000 for Flask
EXPOSE 5000

# Run the Flask app
CMD ["python", "app.py"]
