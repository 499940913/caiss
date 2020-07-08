//
// Created by Chunel on 2020/7/1.
//

#include <iostream>
#include <string>
#include "UtilsInclude.h"
#include <omp.h>
#include <vector>
#include <chrono>

using namespace std;


CAISS_RET_TYPE normalizeNode(std::vector<CAISS_FLOAT>& node) {

    int dim = node.size();

    CAISS_FLOAT sum = 0.0;
    for (unsigned int i = 0; i < dim; i++) {
        sum += (node[i] * node[i]);
    }

    CAISS_FLOAT denominator = std::sqrt(sum);    // 分母信息
    for (unsigned int i = 0; i < dim; i++) {
        node[i] = node[i] / denominator;
    }

    return CAISS_RET_OK;
}

int main() {
    vector<float> vec1;
    vector<float> vec2;

    for (int i = 0; i < 768; i++) {
        vec1.push_back(rand());
        vec2.push_back(rand());
    }

    normalizeNode(vec1);
    normalizeNode(vec2);

    int dim = vec1.size();
    float sum = 0;
#pragma omp parallel for num_threads(4) reduction(+:sum)
    for (int i = 0; i < dim ; i++) {
        sum = vec1[i] * vec2[i];
    }

    cout << 1 - sum << endl;
    return 0;
}