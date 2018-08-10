#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include "hdf5.h"
#include "hdf5_hl.h"

class Output
{
public:
    std::string filename;
    hid_t fid;
    hsize_t dims;
    Output(std::string filename): filename(filename) {
        /* create a HDF5 file */
        std::cout << "# creating a new hdf5 file : " << filename << std::endl;
        fid = H5Fcreate (filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    }

    ~Output() { 
        std::cout << "# closing the hdf5 file and delete output object."  << std::endl;
        H5Fclose(fid);
    };
};

int main(int argc, char** argv) 
{
    const int RANK=2;
    hid_t fid;
    hsize_t dims[RANK] = {2,3};
    std::string filename;
    std::cout << "# filename : ";
    std::cin >> filename;

    auto myoutput = std::make_shared<Output>(filename);
    //auto myoutput = new Output(filename);

    fid = myoutput->fid;

    std::vector<int> data = {2,2,3,4,5,6};

    /* create and write an integer type dataset named "dset" */
    H5LTmake_dataset(fid,"/dset",RANK,dims,H5T_NATIVE_INT,&data[0]);

    //delete myoutput;

    return 0;
}