<b>non_blocking</b>: 

<b>True</b>:
to() method will attempt to perform the data transfer (e.g. of images to GPU) asynchronously,
meaning that the method will return immediately, and the program will not wait for the transfer to
complete before executing the next instruction.
This can lead to a performance improvement, especially with a large data transfer.

<b>False</b>:
program waits for transfer to complete.
Behaviour of *non-blocking* depends on backend: it has no effect on CPUs since they cannot perform async data transfer.
CUDA can do async transfer, and its capabilities depend on the specific version and hardware, as well as other factors.
--- 
<b>ECG data extraction and handling:</b>
currently using convolutional smoother for smoothing the ECG, and pywt lib for applying the wavelet transform.
Check scipy and other libraries for faster methods of doing this.
