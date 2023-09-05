# FROM python:3.11.5-slim-bookworm
FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

# Install project dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# RUN pip install -r requirements.txt
# RUN pip install flask
# RUN pip install requests
# RUN pip install numpy
# RUN pip install scikit-learn
# RUN pip install pickle-mixin
# RUN pip install pandas
# RUN pip install matplotlib
# RUN pip install seaborn
# RUN pip install ppscore

# Copy the project code to the working directory
COPY . .

EXPOSE 5000

# CMD flask --app main run --host 0.0.0.0:5000
# CMD ["flask", "--app", "main", "run", "--host", "0.0.0.0:5000"]
CMD ["flask", "--app", "main", "run", "--host", "0.0.0.0", "--port", "5000"]
