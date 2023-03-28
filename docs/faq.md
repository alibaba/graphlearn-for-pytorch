# Frequently Asked Questions(FAQ)

1. Bus Error

GLT uses shared memory for sharing data between processes. If there is a bus error,
make sure your memory can hold the data and you need to set the shared memory to a
larger size, e.g 128G:
```shell
echo 128000000000 > /proc/sys/kernel/shmmax
mount -o remount,size=128G /dev/shm
```