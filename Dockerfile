# Use Python 3.10 slim image
FROM python:3.10-slim

# Prevent Python from writing pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY web_app/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install --no-cache-dir gunicorn

# Copy full project
COPY web_app/ .

# Collect static files (safe if settings configured for production)
RUN python manage.py collectstatic --noinput || true

# Expose port 9000
EXPOSE 9000

# Run the Django app using gunicorn
CMD ["gunicorn", "floorplan_project.wsgi:application", "--bind", "0.0.0.0:9000"]
