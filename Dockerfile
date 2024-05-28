# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

# Make port 5000 available to the world outside this container
EXPOSE 8080

ENV OPENAI_API_KEY=sk-proj-ZDINUmeoDvJo6WNibHv0T3BlbkFJdldHxoxC6gWTGIez9kYW
ENV GROQ_API_KEY=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
ENV FLASK_ENV=development
ENV FLASK_APP=llm_backend.py
ENV PORT=8080

# Define environment variable
# ENV MODEL_NAME=gpt-3.5-turbo

# Run llm_backend.py when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=8080"]