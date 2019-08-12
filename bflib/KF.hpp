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

    public:
        typedef MatNx1 State;
        typedef MatMx1 Input;
        typedef MatPx1 Output;
        typedef Input Control;
        typedef Output Sensor;
        typedef MatNxN ModelCovariance;
        typedef MatPxP SensorCovariance;
        typedef MatNxN StateMatrix;
        typedef MatNxM InputMatrix;
        typedef MatPxN OutputMatrix;
        
    private:
        typedef void (*ProcessFunction)(StateMatrix &A, InputMatrix &B, OutputMatrix &C, double dt);

        std::default_random_engine gen;
        std::normal_distribution<double> distr{0.0, 1.0};
        std::chrono::time_point<std::chrono::high_resolution_clock> start;

        State x;
        StateMatrix A;
        ModelCovariance Q;
        InputMatrix B;
        OutputMatrix C;
        SensorCovariance R;
        
        MatNx1 randX;
        MatPx1 randY;

        MatNxN P;
        MatNxN Qsqrt;
        MatPxP Rsqrt;

        Output z, yError;
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

        KF(State X) : x(X)
        {
            Q.setIdentity();
            R.setIdentity();
            init();
        }

        KF(ModelCovariance Q, SensorCovariance R) : Q(Q), R(R)
        {
            x.setZero();
            init();
        }

        KF(State X, ModelCovariance Q, SensorCovariance R) : x(X), Q(Q), R(R)
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

        State state()
        {
            State x;
            x.setZero();
            return x;
        }

        Input input()
        {
            Input u;
            u.setZero();
            return u;
        }

        Output output()
        {
            Output y;
            y.setZero();
            return y;
        }

        ModelCovariance createQ()
        {
            ModelCovariance Q;
            Q.setZero();
            return Q;
        }

        SensorCovariance createR()
        {
            SensorCovariance R;
            R.setZero();
            return R;
        }

        void setQ(ModelCovariance Q)
        {
            this->Q = Q;
            P = Q;
            Qsqrt = Q.cwiseSqrt();
        }

        void setR(SensorCovariance R)
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

        virtual void process(StateMatrix &A, InputMatrix &B, OutputMatrix &C, double dt)
        {

        }

        void simulate(State &x, Output &y, Input &u, double dt)
        {
            doProcess(dt);

            randn(randX);
            randn(randY);

            x = A * x + B * u + Qsqrt * randX;
            y = C * x + Rsqrt * randY;
        }

        void run(State &xK, Output &y, Input &u, double dt)
        {
            predict(u, dt);
            update(y);
            xK = x;
        }

    private:
        void predict(Input &u, double dt)
        {
            doProcess(dt);
            x = A * x + B * u;
            P = A * P * A.transpose() + Q;
        }

        void update(Output &y)
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