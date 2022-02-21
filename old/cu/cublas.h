#if !defined CUBLAS_H
#define CUBLAS_H

#include <exception>
#include <cublas_v2.h>


namespace cublas {
  class Error : public std::exception {
    public:
      Error(cublasStatus_t result)
      :
        _result(result)
      {
      }

      virtual const char *what() const noexcept;

      operator cublasStatus_t () const
      {
	return _result;
      }

    private:
      cublasStatus_t _result;
  };


  inline void checkCublasCall(cublasStatus_t result)
  {
    if (result != CUBLAS_STATUS_SUCCESS)
      throw Error(result);
  }


  inline void getMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
  {
    checkCublasCall(cublasGetMatrix(rows, cols, elemSize, A, lda, B, ldb));
  }

  inline void setMatrix(int rows, int cols, int elemSize, const void *A, int lda, void *B, int ldb)
  {
    checkCublasCall(cublasSetMatrix(rows, cols, elemSize, A, lda, B, ldb));
  }


  class Handle {
    public:
      Handle()
      {
	checkCublasCall(cublasCreate(&handle));
      }

      ~Handle()
      {
	checkCublasCall(cublasDestroy(handle));
      }

      operator cublasHandle_t() const
      {
        return handle;
      }

      void setMathMode(cublasMath_t mode)
      {
	checkCublasCall(cublasSetMathMode(handle, mode));
      }

      void gemm
      (
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const void *alpha,
	const void *A, cudaDataType_t Atype, int lda,
	const void *B, cudaDataType_t Btype, int ldb,
	const void *beta,
	void *C, cudaDataType_t Ctype, int ldc,
	cudaDataType_t computeType, cublasGemmAlgo_t algo)
      {
	checkCublasCall(cublasGemmEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, B, Btype, ldb, beta, C, Ctype, ldc, computeType, algo));
      }

      void gemmStridedBatched
      (
	cublasOperation_t transa, cublasOperation_t transb,
	int m, int n, int k,
	const void *alpha,
	const void *A, cudaDataType_t Atype, int lda, long long strideA,
	const void *B, cudaDataType_t Btype, int ldb, long long strideB,
	const void *beta,
	void *C, cudaDataType_t Ctype, int ldc, long long strideC,
	int batchCount, cudaDataType_t computeType, cublasGemmAlgo_t algo
      )
      {
	checkCublasCall(cublasGemmStridedBatchedEx(handle, transa, transb, m, n, k, alpha, A, Atype, lda, strideA, B, Btype, ldb, strideB, beta, C, Ctype, ldc, strideC, batchCount, computeType, algo));
      }

      void gemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const float *alpha, const float *A, int lda, const float *B, int ldb, const float *beta, float *C, int ldc)
      {
	checkCublasCall(cublasSgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
      }

      void gemm(cublasOperation_t transa, cublasOperation_t transb, int m, int n, int k, const __half *alpha, const __half *A, int lda, const __half *B, int ldb, const __half *beta, __half *C, int ldc)
      {
	checkCublasCall(cublasHgemm(handle, transa, transb, m, n, k, alpha, A, lda, B, ldb, beta, C, ldc));
      }

    private:
      cublasHandle_t handle;
  };

}

#endif
