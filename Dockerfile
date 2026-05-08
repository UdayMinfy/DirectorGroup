FROM public.ecr.aws/lambda/python:3.12

# Install dependencies
COPY src/requirements.txt ./
RUN python3 -m pip install -r requirements.txt --target "${LAMBDA_TASK_ROOT}"

# Copy source files
COPY src/ ${LAMBDA_TASK_ROOT}

# Set the Lambda handler
CMD ["app.lambda_handler"]