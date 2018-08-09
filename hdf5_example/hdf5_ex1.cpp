#include <iostream>
#include <string>
#include <vector>
#include "hdf5.h"
#include "hdf5_hl.h"

int main(int argc, char** argv) 
{
    const int RANK=2;
    hid_t fid;
    hsize_t dims[RANK] = {2,3};
    std::string filename;
    std::cout << "# filename :";
    std::cin >> filename;
    
    /* create a HDF5 file */
    fid = H5Fcreate (filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);

    std::vector<int> data = {2,2,3,4,5,6};

    /* create and write an integer type dataset named "dset" */
    H5LTmake_dataset(fid,"/dset",RANK,dims,H5T_NATIVE_INT,&data[0]);

    /* close file */
    H5Fclose (fid);

    return 0;
}