FROM python:3.6-slim

# Install required python packages
RUN pip install --no-cache-dir \
        numpy \
        jupyter \
	pandas \
	scipy \
	sklearn \
	matplotlib \
	torch \
        ;

# Use /work as the working directory
RUN mkdir -p /work
COPY * /work/
WORKDIR /work

# Include the notebook
ADD mini-project1.ipynb /work/mini-project1.ipynb

# Setup jupyter notebook as the default command
# This means that jupyter notebook is launched by default when doing `docker run`.
# Options:
#   --ip=0.0.0.0 bind on all interfaces (otherwise we cannot connect to it)
#   --allow-root force jupyter notebook to start even if we run as root inside the container
#   --NotebookApp.default_url=/notebooks/Assignment_DataScience_Lab_week4.ipynb Open the notebook by default
CMD [ "jupyter", "notebook", "--ip=0.0.0.0", "--allow-root", "--NotebookApp.default_url=/notebooks/mini-project1.ipynb" ]

# Declare port 8888
EXPOSE 8888
