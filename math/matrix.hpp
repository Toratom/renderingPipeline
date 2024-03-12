#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <ostream>
#include <cmath>

template<typename T, size_t N, size_t M>
class Matrix;

template <typename T, size_t N>
using Vector = Matrix<T, N, 1>;


//Declaration of the general class and its specialization (vectors and square matrix)
template <typename T, size_t N, size_t M>
class Matrix {
private:
    T data[N * M] = {0};
    size_t to_idx(size_t i, size_t j) const {return i*M + j;};
public:
    Matrix() {}
    Matrix(const T (&init)[N*M]) {
        for (size_t idx = 0; idx < N*M; idx += 1) {
            data[idx] = init[idx];
        }
    }
    T& operator()(size_t i, size_t j) {
        //Don't need to have T operator()... to do float a = mat(0,0). I guess there is an operator to convert T& to T
        return data[to_idx(i, j)];
    }
    const T& operator()(size_t i, size_t j) const {
        //A const version is needed, for instance for the << operator
        return data[to_idx(i, j)];
    }
    Matrix<T, N, M>& operator+=(const Matrix<T, N, M>& rhs) {
        //Add to itself another matrix of the same size: rhs, the return matrix is itself modified (for chaining)
        for (size_t i = 0; i < N; i += 1) {
            for (size_t j = 0; j < M; j += 1) {
                (*this)(i, j) += rhs(i, j); //this is a pointer to iself, so *this is the object that called the operator +=
            }
        }
        return (*this);
    }
    Matrix<T, N, M>& operator*=(T scalar) {
        for (size_t i = 0; i < N; i += 1) {
            for (size_t j = 0; j < M; j += 1) {
                (*this)(i, j) *= scalar;
            }
        }
        return (*this);
    }
    Matrix<T, N, M>& operator*=(const Matrix<T, M, M>& rhs) {
        (*this) = (*this) * rhs;
        return (*this);
    }
    Matrix<T, N, M>& operator/=(T scalar) {
        (*this) *= (T)(1) / scalar;
        return (*this);
    }
    T dot(const Matrix<T, N, M>& rhs) const {
        T result = 0;
        for (size_t i = 0; i < N; i += 1) {
            for (size_t j = 0; j < M; j += 1) {
                result += (*this)(i, j) * rhs(i, j);
            } 
        }
        return result;
    }

    //For col vectors only
    //see https://stackoverflow.com/questions/13401716/selecting-a-member-function-using-different-enable-if-conditions (an alternative would be a base class see https://www.learncpp.com/cpp-tutorial/partial-template-specialization/)
    template<size_t dummy_M = M> //SFINAE only works as the resolution of the subsitution, needs dummy to have a new subsitution
    T& operator()(size_t i,
                  typename std::enable_if<dummy_M==1, bool>::type=true) {
        return (*this)(i, 0);
    }
    template<size_t dummy_M = M> 
    const T& operator()(size_t i,
                        typename std::enable_if<dummy_M==1, bool>::type=true) const {
        return (*this)(i, 0);
    }
    //Passing array of fixed sized by reference https://stackoverflow.com/questions/5724171/passing-an-array-by-reference
    template<size_t dummy_M = M, size_t K>
    Matrix<T, K, 1> pull(const size_t (&idx)[K],
                         typename std::enable_if<dummy_M==1, bool>::type=true) const {
        Matrix<T, K, 1> result;
        for (size_t i = 0; i < K; i += 1) {
            result(i) = (*this)(idx[i]);
        }
        return result;
    }
    template<size_t dummy_M = M, size_t dummy_N = N> 
    Matrix<T, 3, 1> cross(const Matrix<T, 3, 1>& rhs,
                          typename std::enable_if<dummy_M==1 && dummy_N==3, bool>::type=true) const {
        Matrix<T, 3, 1> result;
        result(0) = (*this)(1)*rhs(2) - (*this)(2)*rhs(1);
        result(1) = (*this)(2)*rhs(0) - (*this)(0)*rhs(2);
        result(2) = (*this)(0)*rhs(1) - (*this)(1)*rhs(0);
        return result;
    }
};


//External operator
template <typename T, size_t N, size_t M>
std::ostream& operator<<(std::ostream& stream, const Matrix<T, N, M>& mat) {
    for (size_t i = 0; i < N; i += 1) {
        for (size_t j = 0; j < M; j += 1) {
            stream << mat(i,j) << " ";
        }
        stream << "\n";
    }
    return stream;
}
template <typename T, size_t N, size_t M>
Matrix<T, N, M> operator+(Matrix<T, N, M> lhs, const Matrix<T, N, M>& rhs) {
    //The lhs is passed by value, so its a copie of the real lhs. The rhs is a const ref (avoid copying data)
    //As lhs is a copy, we can modify its values by using the operator +=
    //The results which is stored by lhs is returne by value, and so will be copy in A in A = B + C
    lhs += rhs;
    return lhs;
}
template <typename T, size_t N, size_t M, size_t K>
Matrix<T, N, M> operator*(const Matrix<T, N, K>& lhs, const Matrix<T, K, M>& rhs) {
    Matrix<T, N, M> result;
    for (size_t i = 0; i < N; i += 1) {
        for (size_t j = 0; j < M; j += 1) {
            for (size_t k = 0; k < K; k += 1) {
                result(i, j) += lhs(i, k) * rhs(k, j);
            }
        }
    }
    return result;
}
template <typename T, size_t N, size_t M>
Matrix<T, N, M> operator*(T scalar, Matrix<T, N, M> rhs) {
    rhs *= scalar;
    return rhs;
}
template <typename T, size_t N, size_t M>
Matrix<T, N, M> operator*(Matrix<T, N, M> lhs, T scalar) {
    lhs *= scalar;
    return lhs;
}
template <typename T, size_t N, size_t M>
Matrix<T, N, M> operator/(T scalar, Matrix<T, N, M> rhs) {
    rhs /= scalar;
    return rhs;
}
template <typename T, size_t N, size_t M>
Matrix<T, N, M> operator/(Matrix<T, N, M> lhs, T scalar) {
    lhs /= scalar;
    return lhs;
}
template <typename T>
Matrix<T, 3, 3> rotation_matrix(const Vector<T, 3>& axis, T theta) {
    T x = axis(0);
    T y = axis(1);
    T z = axis(2);
    T c = cos(theta);
    T s = sin(theta);
    T one = (T)(1);
    Matrix<T, 3, 3> result({(one - c)*x*x + c  , (one - c)*x*y - s*z, (one - c)*x*z - s*y,
                            (one - c)*x*y + s*z, (one - c)*y*y + c  , (one - c)*y*z - s*x,
                            (one - c)*x*z - s*y, (one - c)*y*z + s*z, (one - c)*z*z + c});
    return result;
}

#endif