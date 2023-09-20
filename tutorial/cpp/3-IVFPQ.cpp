/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <iostream>
#include <thread>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/invlists/DirectMap.h>

using idx_t = faiss::idx_t;

int main() {
    long d = 768;      // dimension
    long nb = 50000; // database size
    long nq = 1000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (long i = 0; i < nb; i++) {
        for (long j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (long i = 0; i < nq; i++) {
        for (long j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    long nlist = 100;
    long k = 4;
    long m = d/2;                       // bytes per vector
    long bits_per_code = 4;             // bits per code
    int sanity_check_num = 5;
    int id_offset = 1000;
    faiss::IndexFlatL2 quantizer(d); // the other index
    //faiss::IndexIVFPQ index(&quantizer, d, nlist, m, bits_per_code);
    faiss::IndexIVFPQFastScan index(&quantizer, d, nlist, m, bits_per_code);
    index.set_direct_map_type(faiss::DirectMap::Hashtable);

    std::cout<<"pre train"<<std::endl;
    index.train(nb, xb);
    std::cout<<"after train"<<std::endl;
    //index.add(nb, xb);
    for(long i = 0; i < nb; i++) {
        idx_t id = i + id_offset;
        index.add_with_ids(1, xb + i * d, &id);
    }
    std::cout<<"after add"<<std::endl;

    std::vector<id_t> moved_list;
    for(int i = 0; i < sanity_check_num; i++) {
        idx_t id = id_offset + i;
        auto list_no = faiss::lo_listno(index.direct_map.get(id));
        auto offset = index.get_list_size(list_no) - 1;
        idx_t moved_id = index.invlists->get_single_id(list_no, offset);
        moved_list.push_back(moved_id);
        std::cout<<"moved id:"<<" "<<moved_id<<std::endl;

        index.remove_ids(faiss::IDSelectorArray(1, &id));
    }

    /*for(long i = 0; i < sanity_check_num; i++) {
        idx_t id = i + nb + 10000;
        index.add_with_ids(1, xb + i * d, &id);
    }*/
    
    
    std::cout<<"after remove"<<std::endl;
    

    { // sanity check
        idx_t* I = new idx_t[k * sanity_check_num];
        float* D = new float[k * sanity_check_num];

        float* xbb = new float[sanity_check_num * d];
        for (int i = 0; i < sanity_check_num; i++) {
            for (int j = 0; j < d; j++)
                xbb[d * i + j] = xb[d * (moved_list[i] - id_offset) + j];
        }

        //index.search(sanity_check_num, xbb, k, D, I);
        index.search(sanity_check_num, xb, k, D, I);

        printf("I=\n");
        for (long i = 0; i < sanity_check_num; i++) {
            for (long j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (long i = 0; i < sanity_check_num; i++) {
            for (long j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    std::cout<<"train done"<<std::endl;
    std::this_thread::sleep_for(std::chrono::seconds(1000));

    
    { // search xq
        auto begin = std::chrono::steady_clock::now();
        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

        index.nprobe = 10;
        index.search(nq, xq, k, D, I);
        auto end = std::chrono::steady_clock::now();
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
        printf("Elapsed time: %ld ms\n", elapsed.count());

        /*printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }*/

        delete[] I;
        delete[] D;
    }

    delete[] xq;

    return 0;
}
