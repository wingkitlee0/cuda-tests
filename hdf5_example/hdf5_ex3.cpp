#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <cstdio>

#include "hdf5.h"
#include "hdf5_hl.h"

class Input
{
public:
    std::string filename;
    hid_t fid;
    Input(std::string filename): filename(filename) {
        /* read a HDF5 file */
        std::cout << "# reading a hdf5 file : " << filename << std::endl;
        // should check if the file has been opened..
        fid = H5Fopen (filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    }

    ~Input() { 
        std::cout << "# closing the hdf5 file and delete output object."  << std::endl;
        H5Fclose(fid);
    };
};

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
    fid = myoutput->fid;

    std::vector<int> data = {2,2,3,4,5,6};

    /* create and write an integer type dataset named "dset" */
    H5LTmake_dataset(fid,"/dset",RANK,dims,H5T_NATIVE_INT,&data[0]);

    /* 
        We read the data back to data_in below.
        1. get the rank
        2. get the dimension
        3. get the data
    */

    size_t     i, j, nrow, n_values;
    int rank_in;

    auto myinput = std::make_shared<Input>(filename);

    /* get the rank of the dataset */
    H5LTget_dataset_ndims( myinput->fid, "/dset", &rank_in);

    std::cout << "# rank of data = " << rank_in << std::endl;

    std::vector<hsize_t> dims_in(rank_in); // create the dims_in array

    /* get the dimensions of the dataset */
    H5LTget_dataset_info( myinput->fid,"/dset",&dims_in[0],NULL,NULL);

    n_values = 1;
    for (auto dim : dims_in) n_values *= (size_t) dim;
    
    std::cout << "# n_values = " << n_values << std::endl;

    std::vector<int> data_in(n_values);

    /* read dataset */
    H5LTread_dataset_int( myinput->fid,"/dset",&data_in[0]);

    /* print it by rows */
    nrow = (size_t)dims[1];
    for (i=0; i<n_values/nrow; i++ )
    {
        for (j=0; j<nrow; j++) printf ("  %d", data_in[i*nrow + j]);
            printf ("\n");
    }
    
    return 0;
}