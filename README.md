Creating the docker image:
```
docker build -t deeplearningproj1 .
```

To run the image:
```
docker run --init --rm -it -p 8888:8888 deeplearningproj1
```
