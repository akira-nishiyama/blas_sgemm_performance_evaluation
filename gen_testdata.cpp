#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <cblas.h>
#include <iomanip>

int main(int argc, char* argv[]){

  if(argc != 4){
    std::cout << "Usage:gen_testdata initial max inc" << std::endl;
    return -1;
  }

  int ini = std::atoi(argv[1]);
  int max = std::atoi(argv[2]);
  int inc = std::atoi(argv[3]);

  for(int i = ini; i <= max; i += inc){
    std::cout << "generating num=" << i << "..." << std::endl;
    int m = i, k = i, n = i;
    float alpha = 1.0f;
    float beta = 0.0f;

    std::string a_name = "a_" + std::to_string(i) + ".txt";
    std::string b_name = "b_" + std::to_string(i) + ".txt";
    std::string c_name = "c_" + std::to_string(i) + ".txt";

    std::vector<float> a(m * k);
    std::vector<float> b(k * n);
    std::vector<float> c(m * n);

    std::ofstream a_out(a_name);
    std::ofstream b_out(b_name);
    std::ofstream c_out(c_name);

    a_out << std::fixed << std::setprecision(15);
    b_out << std::fixed << std::setprecision(15);
    c_out << std::fixed << std::setprecision(15);

    std::mt19937 mt(i);
    std::uniform_real_distribution<float> dist(0.0,1.0);

    std::cout << "gen a..." << std::endl;
    for(auto &d: a){
      d = dist(mt);
      a_out << d << std::endl;
    }
    std::cout << "gen b..." << std::endl;
    for(auto &d: b){
      d = dist(mt);
      b_out << d << std::endl;
    }
    std::cout << "calc c..." << std::endl;
    cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1.0, a.data(), k, b.data(), n, 0.0, c.data(), n);

    std::cout << "write c..." << std::endl;
    for(float &d: c){
      c_out << d << std::endl;
    }

    a_out.close();
    b_out.close();
    c_out.close();
    std::cout << "done!" << std::endl;

  }

  return 0;
}
