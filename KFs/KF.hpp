#pragma once

#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <vector>
#include <thread>

using namespace Eigen;

template <typename dataType, int states, int inputs, int outputs>
class KF
{
    private:
        typedef Matrix<dataType, states, 1> MatNx1;
        typedef Matrix<dataType, states, states> MatNxN;
        typedef Matrix<dataType, states, inputs> MatNxM;
        typedef Matrix<dataType, states, outputs> MatNxP;
        typedef Matrix<dataType, outputs, states> MatPxN;
        typedef Matrix<dataType, outputs, outputs> MatPxP;
        typedef Matrix<dataType, inputs, 1> MatMx1;
        typedef Matrix<dataType, outputs, 1> MatPx1;
        typedef void (*ProcessFunction)(MatNxN &A, MatNxM &B, MatPxN &C, double dt);

        std::default_random_engine gen;
        std::normal_distribution<double> distr{0.0, 1.0};
        std::chrono::time_point<std::chrono::high_resolution_clock> start;

        MatNx1 x, randX;
        MatNxN A, Q, P;
        MatNxM B;
        MatPxN C;
        MatPxP R;
        MatPx1 randY;

        MatNxN Qsqrt;
        MatPxP Rsqrt;

        MatPx1 z, yError;
        MatPxP S;
        MatNxP K;
        MatNxN I;


        ProcessFunction processFn;

        void init()
        {
            processFn = NULL;
            P = Q;
            I.setIdentity();

            Qsqrt = Q.cwiseSqrt();
            Rsqrt = R.cwiseSqrt();

            start = std::chrono::high_resolution_clock::now();
        }
    public:
        KF()
        {
            Q.setIdentity();
            R.setIdentity();
            x.setZero();
            init();
        }

        KF(MatNx1 X) : x(X)
        {
            Q.setIdentity();
            R.setIdentity();
            init();
        }

        KF(MatNxN Q, MatPxP R) : Q(Q), R(R)
        {
            x.setZero();
            init();
        }

        KF(MatNx1 X, MatNxN Q, MatPx1 R) : x(X), Q(Q), R(R)
        {
            init();
        }

        virtual ~KF()
        {

        }

        void seed()
        {
            gen = std::default_random_engine(std::chrono::system_clock::now().time_since_epoch().count());
        }

        void seed(long long s)
        {
            gen = std::default_random_engine(s);
        }

        MatNx1 state()
        {
            MatNx1 x;
            x.setZero();
            return x;
        }

        MatMx1 input()
        {
            MatMx1 u;
            u.setZero();
            return u;
        }

        MatPx1 output()
        {
            MatPx1 y;
            y.setZero();
            return y;
        }

        MatNxN createQ()
        {
            MatNxN Q;
            Q.setZero();
            return Q;
        }

        MatPxP createR()
        {
            MatPxP R;
            R.setZero();
            return R;
        }

        void setQ(MatNxN Q)
        {
            this->Q = Q;
            P = Q;
            Qsqrt = Q.cwiseSqrt();
        }

        void setR(MatPxP R)
        {
            this->R = R;
            Rsqrt = R.cwiseSqrt();
        }

        double time()
        {
            auto end = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> diff = end - start;
            start = std::chrono::high_resolution_clock::now();
            return diff.count();
        }

        double delay(double s)
        {
            double ellapsed = time();
            double remain = s - ellapsed;
            if(remain < 0)
                return ellapsed;
            std::this_thread::sleep_for(std::chrono::nanoseconds((long long)(remain * 1e9)));
            ellapsed += time();
            return ellapsed;
        }

        void setProcess(ProcessFunction fn)
        {
            processFn = fn;
        }

        virtual void process(MatNxN &A, MatNxM &B, MatPxN &C, double dt)
        {

        }

        void simulate(MatNx1 &x, MatPx1 &y, MatMx1 &u, double dt)
        {
            doProcess(dt);

            randn(randX);
            randn(randY);

            x = A * x + B * u + Qsqrt * randX;
            y = C * x + Rsqrt * randY;
        }

        void run(MatNx1 &xK, MatPx1 &y, MatMx1 &u, double dt)
        {
            predict(u, dt);
            update(y);
            xK = x;
        }

    private:
        void predict(MatMx1 &u, double dt)
        {
            doProcess(dt);
            x = A * x + B * u;
            P = A * P * A.transpose() + Q;
        }

        void update(MatPx1 &y)
        {
            z = C * x;
            yError = y - z;

            S = C * P * C.transpose() + R;
            K = P * C.transpose() * S.inverse();
            x = x + K * yError;

            P = (I - K * C) * P;
        }

        void doProcess(double dt)
        {
            if(processFn != NULL)
                processFn(A, B, C, dt);
            else
                process(A, B, C, dt);
        }

        template<class T>
        void randn(T &mat)
        {
            for (size_t i = 0; i < mat.rows(); i++)
            {
                for (size_t j = 0; j < mat.cols(); j++)
                {
                    mat(i, j) = distr(gen);
                }
            }
        }

};