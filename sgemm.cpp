#include <iostream>
#include <fstream>
#include <chrono>
#include <cblas.h>
#include <vector>
#include <cmath>
//#include <random>
#ifdef CUBLAS
#include <cuda_runtime.h>
#include "cublas_v2.h"
#endif

const std::string log_file = "exec_result.csv";

int main(int argc, char **argv){
  //std::cout << "type,num,duration(ms),diff_sum" << std::endl;
  std::string type;
  #ifdef OPENBLAS
    type = "OPENBLAS";
  #elif CUBLAS
    type = "CUBLAS";
    cublasHandle_t test_h; 
    cublasCreate(&test_h);
    cublasDestroy(test_h);
  #elif OPENMP
    type = "OPENMP";
  #else
    type = "C_LOOP";
  #endif

  if(argc != 3){
    std::cout << "Usage: ./sgemm_* array_size loop_num" << std::endl;
    std::cout << "array_size:[1,10,100,1000,10000]" << std::endl;
    std::cout << "loop_num:1-" << std::endl;
    return -1;
  }
  int num = std::atoi(argv[1]);
  int lop = std::atoi(argv[2]);

  std::ofstream h_log_file(log_file,std::ios::app);

  while(lop){
    int m = num, k = num, n = num;
    float alpha = 1.0f;
    float beta = 0.0f;

    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c(m * n);
    std::vector<float> c_golden(m * n);

    std::string a_name = "a_" + std::to_string(num) + ".txt";
    std::string b_name = "b_" + std::to_string(num) + ".txt";
    std::string c_name = "c_" + std::to_string(num) + ".txt";

    std::ifstream a_in(a_name);
    std::ifstream b_in(b_name);
    std::ifstream c_in(c_name);

    if(a_in.fail() || b_in.fail() || c_in.fail()){
      std::cerr << "failed to open test data file." << std::endl;
      return -1;
    }

    for(int j = 0; j < m*k; ++j){
      a_in >> a[j];
    }
    for(int j = 0; j < k*n; ++j){
      b_in >> b[j];
    }
    for(int j = 0; j < m*n; ++j){
      c[j] = 0;
    }
    for(int j = 0; j < m*k; ++j){
      c_in >> c_golden[j];
    }

    //std::mt19937 mt(i);
    //std::uniform_real_distribution<float> dist(0.0,10.0);

    //for(auto &d: a){
    //  d = dist(mt);
    //}
    //for(auto &d: b){
    //  d = dist(mt);
    //}

    auto start_time = std::chrono::system_clock::now();

#ifdef OPENBLAS
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a.data(), k, b.data(), n, 0.0, c.data(), n);

#elif CUBLAS

    //デバイス(GPU)側用 　
    float *devA,*devB,*devC;
    
    cudaMalloc((void **)&devA,(m * k * sizeof(float)));
    cudaMalloc((void **)&devB,(k * n * sizeof(float)));
    cudaMalloc((void **)&devC,(m * n * sizeof(float)));

    cublasSetVector(m*k, sizeof(float), a.data(), 1, devA, 1);
    cublasSetVector(k*n, sizeof(float), b.data(), 1, devB, 1);

    cublasHandle_t handle; 
    cublasCreate(&handle);

    cublasSgemm(
      handle,
      CUBLAS_OP_N, //行列A 転置有無
      CUBLAS_OP_N, //行列B 転置有無
      m,    // 行列Aの行数
      n,    // 行列Bの列数
      k,    // 行列Aの列数(=行列Ｂの行数)
      &alpha, // 行列の積に掛ける値(なければ1)
      devA,   // 行列A
      m,    // 行列Aの行数
      devB,   // 行列B
      k,    // 行列Bの行数
      &beta,  // 行列Cに掛けるスカラ値(なければ0)]
      devC,   // 行列Cの初期値 兼 出力先
      m // 行列Cの行数
    );

    auto status = cublasDestroy(handle);
    
    cublasGetVector(m*n, sizeof(float), devC, 1, c.data(), 1);

    cudaFree(devA);
    cudaFree(devB);
    cudaFree(devC);

#else
#pragma omp parallel for
    for (int i = 0; i < n; ++i) {
      for (int j = 0; j < n; ++j) {
        for (int k = 0; k < n; ++k) {
          c[j * n + i] += a[k * n + i] * b[j * n + k];
        }
      }
    }
#endif


    auto end_time = std::chrono::system_clock::now();

    cblas_saxpy(c.size(),-1.0,c_golden.data(),1.0,c.data(),1.0);

    float diff=0;
    for(int i = 0; i < m; ++i){
      for(int j = 0; j < n; ++j){
        if(c[i*n+j] > std::numeric_limits<float>::epsilon() * c_golden[i*n+j] * std::sqrt(n)){
          diff += c[i*n+j];
        }
      }
    }

    auto duration = std::chrono::duration_cast<
                      std::chrono::milliseconds>(end_time - start_time).count();
    
    std::cout << type << "," << num << "," << duration << "," << diff << std::endl;
    h_log_file << type << "," << num << "," << duration << "," << diff << std::endl;
    --lop;
  }
  return 0;
}
