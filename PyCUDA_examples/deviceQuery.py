import pycuda.autoinit
from pycuda import driver

CUDA_CORES_PER_MP = { 5.0 : 128, 5.1 : 128, 5.2 : 128,
                        6.0 : 64, 6.1 : 128, 6.2 : 128}



for i in range(driver.Device.count()):
    gpu = driver.Device(i)
    print("Device {}: {}".format(i, gpu.name() ))
    compute_capability = float( "%d.%d" % gpu.compute_capability() )

    print("Compute Capability: {}".format(compute_capability))
    total_memory = gpu.total_memory() // (1024**2)
    print("Total Memory: {} MB".format(total_memory))

    #device_attributes_tuples = gpu.get_attributes().items()
    #print(type(device_attributes_tuples))
    device_attributes = dict( (str(k),v) for k, v in gpu.get_attributes().items())
    #print(device_attributes)

    num_mp = device_attributes['MULTIPROCESSOR_COUNT']
    cuda_cores_per_mp = CUDA_CORES_PER_MP[compute_capability]

    print("\t ({}) MP, ({}) CUDA Cores/MP: {} CUDA Cores".format(num_mp, cuda_cores_per_mp, num_mp*cuda_cores_per_mp))

    device_attributes.pop('MULTIPROCESSOR_COUNT')
    for k, v in device_attributes.items():
        print("\t {}: {}".format(k, v))