FROM python:3.11-slim

# Set work directory
WORKDIR /app

# Install system dependencies including git
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 libsm6 libxext6 libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the full project code
COPY . .

# Expose port (if using Django default)
EXPOSE 8000

# Command to run Django app
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
