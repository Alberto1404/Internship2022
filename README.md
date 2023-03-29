# Internship2022

Mention to https://github.com/jocpae/clDice for Centerline loss and metric required for this project.
Run using SLURM and Singularity. 

1. Create an empty singularity docker
``` sh
$ singularity pull docker://ubuntu
```
2. Once done, all libraries must be installed. To pursue that, we shall convert .SIF to SANDBOX, open it, and reconvert to SIF again, depicted in the following commands:
``` sh
# SIF_FILE (SIF--> SANDBOX)
$ sudo singularity build --sandbox SANDBOX  

# (Open and write in SANDBOX (ex. install packages) )
$ sudo singularity shell --writable SANDBOX/ 

# (SANDBOX-->SIF)
$ sudo singularity build SIF_FILE SANDBOX

```




That's all folks!
