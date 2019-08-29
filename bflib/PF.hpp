
// Copyright(c) 2019-present, Alexander Silva Barbosa & bflib contributors.
// Distributed under the MIT License (http://opensource.org/licenses/MIT)

/**
 * @author Alexander Silva Barbosa <alexander.ti.ufv@gmail.com>
 * @date 2019
 * Particle Filter (Monte Carlo)
 */

#pragma once

#include <Eigen/Dense>
#include <random>
#include <chrono>
#include <thread>
#include <vector>

using namespace Eigen;

template <typename dataType, int states, int inputs, int outputs, int outputSize>
class PF
{
    private:
        typedef Matrix<dataType, states, 1> MatNx1;
        typedef Matrix<dataType, inputs, 1> MatMx1;
        typedef Matrix<dataType, outputs, 1> MatPx1;
        typedef Matrix<dataType, states, states> MatNxN;

    public:
        typedef MatNx1 State;
        typedef MatMx1 Input;
        typedef MatPx1 Output;
        typedef Input Control;
        typedef Output Sensor;
        typedef MatPx1 SensorStd;
        typedef MatNx1 ResampleStd;

        double eps;

    private:
        typedef void (*ModelFunction)(State &x, Input &u, double dt);
        typedef void (*SensorFunction)(std::vector<Output> &y, State &x, double dt);

        std::default_random_engine gen;
        std::normal_distribution<double> distr{0.0, 1.0};
        std::chrono::time_point<std::chrono::high_resolution_clock> start;

        int N, NMax;
        std::vector<State> W;
        std::vector< std::vector<Output> > Y;
        std::vector<double> w;
        double wMax;

        State minState, maxState, meanState;

        State x;
        ResampleStd Q;
        SensorStd R;
    
        MatNxN resampleMatrix;
        MatNx1 randX;

        ModelFunction modelFn;
        SensorFunction sensorFn;

        void init()
        {
            modelFn = NULL;
            sensorFn = NULL;

            resampleMatrix = Q.asDiagonal();

            N = NMax;
            W.resize(N);
            Y.resize(N);
            w.resize(N);
            meanState = (maxState - minState) / 2;
            initW();

            eps = 1e-10;

            start = std::chrono::high_resolution_clock::now();
        }
    public:

        PF(int N, State minState, State maxState) : NMax(N), minState(minState), maxState(maxState)
        {
            Q.setIdentity();
            R.setIdentity();
            x.setZero();
            init();
        }

        PF(int N, State X, State minState, State maxState) : x(X), NMax(N), minState(minState), maxState(maxState)
        {
            Q.setIdentity();
            R.setIdentity();
            init();
        }

        PF(int N, State minState, State maxState, ResampleStd Q, SensorStd R) : Q(Q), R(R), NMax(N), minState(minState), maxState(maxState)
        {
            x.setZero();
            init();
        }

        PF(int N, State X, State minState, State maxState, ResampleStd Q, SensorStd R) : x(X), Q(Q), R(R), NMax(N), minState(minState), maxState(maxState)
        {
            init();
        }

        virtual ~PF()
        {

        }

        void seed()
        {
            unsigned int s = std::chrono::system_clock::now().time_since_epoch().count();
            seed(s);
        }

        void seed(unsigned int s)
        {
            gen = std::default_random_engine(s);
            srand (s);
            std::srand(s);
            initW();
        }

        State state()
        {
            MatNx1 x;
            x.setZero();
            return x;
        }

        Input input()
        {
            MatMx1 u;
            u.setZero();
            return u;
        }

        Output output()
        {
            MatPx1 y;
            y.setZero();
            return y;
        }

        SensorStd createR()
        {
            SensorStd R;
            R.setZero();
            return R;
        }

        void setR(SensorStd R)
        {
            this->R = R;
        }

        ResampleStd createQ()
        {
            ResampleStd Q;
            Q.setZero();
            return Q;
        }

        void setQ(ResampleStd Q)
        {
            this->Q = Q;
            resampleMatrix = Q.asDiagonal();
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

        std::vector<State>& particles()
        {
            return W;
        }

        void setModel(ModelFunction fn)
        {
            modelFn = fn;
        }

        void setSensor(SensorFunction fn)
        {
            sensorFn = fn;
        }

        virtual void model(State &x, Input &u, double dt)
        {

        }

        virtual void sensor(std::vector<Output> &z, State &x, double dt)
        {

        }

        void simulate(State &x, std::vector<Output> &y, Input &u, double dt)
        {
            doModel(x, u, dt);

            y.resize(outputSize);
            doSensor(y, x, dt);
        }

        void run(State &xK, std::vector<Output> &y, Input &u, double dt)
        {
            y.resize(outputSize);

            applyControl(u, dt);
            sense(dt);
            compare(y);
            resample();
            measure();

            xK = x;
        }

    private:
        void initW()
        {
            for(int n = 0; n < N; n++)
            {
                W[n] = State::Random();
                for (size_t i = 0; i < states; i++)
                {
                    W[n](i) = meanState(i) + W[n](i) * meanState(i);
                }
            }
        }

        void applyControl( Input &u, double dt)
        {
            for(int i = 0; i < N; i++)
            {
                doModel(W[i], u, dt);
            }
        }

        void sense(double dt)
        {
            for(int i = 0; i < N; i++)
            {
                Y[i].resize(outputSize);
                doSensor(Y[i], W[i], dt);
            }
        }

        void compare(std::vector<Output> &y)
        {
            double sum = 0;
            double lh = 0;
            for(int i = 0; i < N; i++)
            {
                w[i] = 1.0;
                for(int j = 0; j < outputSize; j++)
                {
                    lh = likelihood(y[j], Y[i][j]);
                    w[i] *= lh;
                }
                sum += w[i];
            }

            wMax = 0;
            for(int i = 0; i < N; i++)
            {
                w[i] /= sum;
                if(w[i] > wMax)
                    wMax = w[i];
            }
        }

        double likelihood(Output y, Output z)
        {
            double lh = 0;
            double df, sq, ex, pi2, m;

            pi2 = sqrt(2 * 3.14159265358979);

            for (size_t i = 0; i < outputs; i++)
            {
                if(R(i) == 0)
                    continue;

                m = 1.0 / ( R(i) * pi2 );

                df = (z(i) - y(i)) / R(i);

                sq = df*df;
                ex = exp( -0.5 * sq );

                lh += m * ex;
            }

            lh += eps;
        }

        void resample()
        {
            int index = rand() % N;
            double F;
            std::vector<State> nW(N);

            for(int i = 0; i < N; i++)
            {
                F = 10 * wMax * ( ( rand() % 100 ) / 100.0);
                while(F > w[index])
                {
                    F -= w[index];
                    index = ( index + 1 ) % N;
                }
                randn(randX);
                nW[i] = W[index] + resampleMatrix * randX;
                for (size_t j = 0; j < outputs; j++)
                {
                    if(nW[i](j) < minState(j))
                        nW[i](j) = maxState(j) - ( minState(j) - nW[i](j) );
                    else if(nW[i](j) > maxState(j))
                        nW[i](j) = minState(j) + ( nW[i](j) - maxState(j) );
                }
            }

            W = nW;
        }

        void measure()
        {
            x.setZero();
            for(int i = 0; i < N; i++)
            {
                x += W[i];
            }
            x /= N;
        }

        void doModel(State &x, Input &u, double dt)
        {
            if(modelFn != NULL)
                modelFn(x, u, dt);
            else
                model(x, u, dt);
        }

        void doSensor(std::vector<Output> &z, State &x, double dt)
        {
            if(sensorFn != NULL)
                sensorFn(z, x, dt);
            else
                sensor(z, x, dt);
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