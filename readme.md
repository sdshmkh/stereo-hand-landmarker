### Stereo Hand landmarker

Utilizes Stereo cameras to dectect handland marks and render in 3D space

## Setting up

### Docker
Install [Docker](https://www.docker.com/). 

### Installing dependancies

We can install dependacies using conda. 

```
conda env create -n <env-name>  -f environment.yml
```

```
conda activate <env-name>
```

## Starting Application

First start Redis using `docker compose`. Make sure you are in the project directory on the terminal. 

```
docker compose up -d
```

This will start the redis container in the background. 

Run 
```
python o3d.py
```

Now in another shell run
```
python main.py
```