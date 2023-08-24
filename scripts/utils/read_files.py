import h5py


def read_file(fname):
    with h5py.File(fname, "r") as f:
        results = {"images": f["images"][:]}

        for name, val in f["images"].attrs.items():
            results[name] = val

    return results
