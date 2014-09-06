#pragma once

template<typename T, int K>
class MathArray
{
  private:
    T data[K];
    template<typename Func>
      static void foreach(Func F) { for (int i = 0; i < K; i++) F(i); }

    template<typename Func>
      static MathArray& foreach(MathArray &res, Func F) { for (int i = 0; i < K; i++) F(i);  return res;}


  public:

    int size() const {return K; }
    MathArray() {}
    explicit MathArray(const T &v) { for (auto &x : data) x = v; }
    T& operator[](const int i)       { return data[i]; }
    T  operator[](const int i) const {return data[i]; }

    MathArray &operator-(MathArray rhs) { return foreach(rhs, [&](int i) {rhs[i] = -rhs[i];}); }

    MathArray& operator+=(const MathArray& rhs) { foreach([&](int i){ data[i] += rhs[i]; }); return *this; }
    MathArray& operator-=(const MathArray& rhs) { foreach([&](int i){ data[i] -= rhs[i]; }); return *this; }
    MathArray& operator*=(const MathArray& rhs) { foreach([&](int i){ data[i] *= rhs[i]; }); return *this; }
    MathArray& operator/=(const MathArray& rhs) { foreach([&](int i){ data[i] /= rhs[i]; }); return *this; }
    friend MathArray operator+(MathArray rhs, const MathArray& lhs) { return (rhs += lhs); }
    friend MathArray operator-(MathArray rhs, const MathArray& lhs) { return (rhs -= lhs); }
    friend MathArray operator*(MathArray rhs, const MathArray& lhs) { return (rhs *= lhs); }
    friend MathArray operator/(MathArray rhs, const MathArray& lhs) { return (rhs /= lhs); }

    MathArray& operator+=(const T& rhs) { return foreach(*this, [&](int i){ data[i] += rhs; }); }
    MathArray& operator-=(const T& rhs) { return foreach(*this, [&](int i){ data[i] -= rhs; }); }
    MathArray& operator*=(const T& rhs) { return foreach(*this, [&](int i){ data[i] *= rhs; }); }
    MathArray& operator/=(const T& rhs) { return foreach(*this, [&](int i){ data[i] /= rhs; }); }
    friend MathArray operator+(MathArray rhs, const T& lhs) { return (rhs += lhs); }
    friend MathArray operator-(MathArray rhs, const T& lhs) { return (rhs -= lhs); }
    friend MathArray operator*(MathArray rhs, const T& lhs) { return (rhs *= lhs); }
    friend MathArray operator/(MathArray rhs, const T& lhs) { return (rhs /= lhs); }

    friend MathArray operator+(const T& rhs, MathArray lhs) { return (lhs += rhs); }
    friend MathArray operator*(const T& rhs, MathArray lhs) { return (lhs *= rhs); }
    friend MathArray operator-(const T& rhs, MathArray lhs) { return foreach(lhs, [&](int i){lhs[i] = rhs - lhs[i];}); }
    friend MathArray operator/(const T& rhs, MathArray lhs) { return foreach(lhs, [&](int i){lhs[i] = rhs / lhs[i];}); }

};
