# Main-Memory Partitioning and Sorting

This repository contains the in-place MSD radix sort implementation of the publication 

``` BibTeX
@inproceedings{polychroniou2014comprehensive,
  title={A comprehensive study of main-memory partitioning and its application to large-scale comparison-and radix-sort},
  author={Polychroniou, Orestis and Ross, Kenneth A},
  booktitle={Proceedings of the 2014 ACM SIGMOD international conference on Management of data},
  pages={755--766},
  year={2014}
}
```

from Orestis Polychroniou et al. The source code has been retrieved from Orestis Polychroniou's personal web page [www.cs.columbia.edu/~orestis](http://www.cs.columbia.edu/~orestis). 

This repository adds support for CMake and provides a library for the in-place MSD radix sort algorithm.

Limitations:

1) All CPUs should have the same number of cores.
   Other configurations will not adapt properly.
2) LSB methods can run for up to 8 sockets
   due to implementing up to 8-way range partition.
3) MSB methods must use exactly 64 threads
   due to implementing only 64-way range partition.
4) MSB methods need a large fudge factor or a small
   block, otherwise the algorithm will be unable to
   distribute blocks across NUMA regions correctly.
5) MSB methods are implemented to run on full
   32-bit data while LSB methods can skip bits.
6) Zipfian distributions are implemented for
   32 bit data only and not for 64-bit data.
