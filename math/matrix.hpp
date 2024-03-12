#ifndef MATRIX_H
#define MATRIX_H

#include <stddef.h>
#include <ostream>
#include <cmath>

template<typename T, size_t N, size_t M>
class Matrix;

template <typename T, size_t N>
using Vector = Matrix<T, N, 1>;

using Mat3 = Matrix<double, 3, 3>;
using Mat4 = Matrix<double, 4, 4>;
using Vec3 = Vector<double, 3>;
using Vec4 = Vector<double, 4>;

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
    Matrix<T, N, M>& operator-=(const Matrix<T, N, M>& rhs) {
        (*this) += (T)(-1)*rhs;
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
    Matrix<T, M, N> transpose() {
       Matrix<T, M, N> result;
       for (size_t i = 0; i < M; i += 1) {
            for (size_t j = 0; j < N; j += 1) {
                result(i, j) = (*this)(j, i);
            }
       }
       return result;
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
    T norm() const {
        T result = (*this).dot(*this);
        return sqrt(result);
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
template <typename T, size_t N, size_t M>
Matrix<T, N, M> operator-(Matrix<T, N, M> lhs, const Matrix<T, N, M>& rhs) {
    lhs -= rhs;
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


//Utils
template <typename T>
Matrix<T, 3, 3> rotation_matrix3(const Vector<T, 3>& axis, T theta) {
    T x = axis(0);
    T y = axis(1);
    T z = axis(2);
    T c = cos(theta);
    T s = sin(theta);
    Matrix<T, 3, 3> result({(1 - c)*x*x + c  , (1 - c)*x*y - s*z, (1 - c)*x*z + s*y,
                            (1 - c)*x*y + s*z, (1 - c)*y*y + c  , (1 - c)*y*z - s*x,
                            (1 - c)*x*z - s*y, (1 - c)*y*z + s*x, (1 - c)*z*z + c});
    return result;
}
template <typename T>
Matrix<T, 4, 4> rotation_matrix(const Vector<T, 3>& axis, T theta) {
    Matrix<T, 3, 3> tmp = rotation_matrix3(axis, theta);
    Matrix<T, 4, 4> result({tmp(0,0), tmp(0,1), tmp(0,2), 0,
                            tmp(1,0), tmp(1,1), tmp(1,2), 0,
                            tmp(2,0), tmp(2,1), tmp(2,2), 0,
                            0       , 0       , 0       , 1});
    return result;
}
template <typename T>
Matrix<T, 4, 4> translation_matrix(const Vector<T, 3>& target) {
    Matrix<T, 4, 4> result({1, 0, 0, target(0),
                            0, 1, 0, target(1),
                            0, 0, 1, target(2),
                            0, 0, 0, 1});
    return result;
}
template <typename T>
Matrix<T, 3, 3> scale_matrix3(T s) {
    Matrix<T, 3, 3> result({s, 0, 0,
                            0, s, 0,
                            0, 0, s});
    return result;
}
template <typename T>
Matrix<T, 4, 4> scale_matrix(T s) {
    Matrix<T, 4, 4> result({s, 0, 0, 0,
                            0, s, 0, 0,
                            0, 0, s, 0,
                            0, 0, 0, 1});
    return result;
}
template <typename T>
Matrix<T, 4, 4> look_at_matrix(const Vector<T, 3>& position, const Vector<T, 3>& target, const Vector<T, 3>& up) {
    Vector<T, 3> Z = position - target;
    Z /= Z.norm();
    Vector<T, 3> X = up.cross(Z);
    Vector<T, 3> Y = Z.cross(X);
    Matrix<T, 4, 4> result({X(0), X(1), X(2), -X.dot(position),
                            Y(0), Y(1), Y(2), -Y.dot(position),
                            Z(0), Z(1), Z(2), -Z.dot(position),
                            0   , 0   , 0   , 1});
    return result;
}
template <typename T>
Matrix<T, 4, 4> projection_matrix(T left, T right, T bottom, T top, T near, T far) {
    Matrix<T, 4, 4> result({2*near/(right - left), 0                    , (right + left)/(right - left), 0,
                            0                    , 2*near/(top - bottom), (top + bottom)/(top - bottom), 0,
                            0                    , 0                    , -(far + near)/(far - near)   , -2*far*near/(far - near),
                            0                    , 0                    , -1                           , 0});
    return result;
}


#endif