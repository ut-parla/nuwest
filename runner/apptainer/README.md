## Scripts for Docker Container Execution

These are assumed to be run from the root of the repository.

#### Run a script

```bash
./runner/docker/run.sh lessons/parla/scripts/01_hello_world.py
```
This will run the script in the container and print the output to the terminal.

The `--use-gpu` flag is available to run on a GPU-enabled container.


#### Profile a script

```bash
./runner/docker/profile.sh lessons/parla/scripts/01_hello_world.py
```
The profile trace will be saved to `reports/<script name>.nsys-rep`.

The `--use-gpu` flag is available to run on a GPU-enabled container.

#### Launch a Jupyter Notebook Server

```
./runner/docker/notebook.sh
```
This will launch a Jupyter Notebook Server on port 8888 with password `NUWEST2024`. 

The `--use-gpu` flag is available to run on a GPU-enabled container.

## Connecting to a remote Jupyter Notebook Server

If you are running the container on a remote machine, you can connect to the Jupyter Notebook Server via SSH tunneling.

```
ssh -L <local_port>:localhost:8888 <username>@<remote machine>
```

Then, open a browser and navigate to `localhost:<local_port>` to access the Jupyter Notebook Server.