#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <sstream>
#include <random>

#include "MpiCalcPi.h"


struct Settings {
    MpiCalcPi::Hits totalNumThrows_{ int(5e6) }; // TOTAL throws at dartboard
};



MpiCalcPi::MpiCalcPi() :
    MpiProcess()
{
}


MpiCalcPi::~MpiCalcPi()
{
}


int
MpiCalcPi::runAsManagerImpl(const StringArray1 &args)
{
    std::cout << getVersionString() << std::endl;

    Settings s;
    int ret = processArgs(args, s);
    if (!MPIOK(ret)) {
        // ret already set
    }
    else if (!MPIOK(MPI_Bcast(&s, sizeof(s), MPI_UNSIGNED_CHAR, managerTaskId(), comm()))) {
        ret = ErrBcast;
    }
    else {
        // Manager task also picks up any throws lost to integer truncation.
        const Hits numThrows = (s.totalNumThrows_ / numTasks()) +
            (s.totalNumThrows_ % (s.totalNumThrows_ / numTasks()));

        // compute pi for this task
        const Hits hits = throwDarts(numThrows);

        std::cout << "Task " << taskId() << " had " << hits <<
            " hits out of " << numThrows << " throws" << std::endl;

        Hits sumHits = 0; // sum of ALL subprocess hits
        if (!MPIOK(MPI_Barrier(comm()))) {
            ret = ErrBarrier;
        }
        else if (!mpiReduceSumHits(hits, sumHits)) {
            ret = ErrReduce;
        }
        else {
            // Manager and all subtasks have computed their values for PI. The
            // call to MPI_Reduce() has summed them all together and placed
            // result into sumHits.
            printf("After %llu throws...\n", s.totalNumThrows_);
            fflush(stdout);
            const double computedPi{ (4.0 * sumHits) / s.totalNumThrows_ };
            const double actualPi{ 3.1415926535897 };
            const double piError{ actualPi - computedPi };
            std::cout << "  Computed PI : " << computedPi << std::endl;
            std::cout << "  Actual   PI : " << actualPi << std::endl;
            std::cout << "  Error       : " << piError << std::endl;
        }
    }
    return ret;
}


int
MpiCalcPi::runAsWorkerImpl(const StringArray1 &args)
{
    int ret = ErrNone;
    Settings s;
    if (!mpiBcast(&s, sizeof(s))) {
        ret = ErrBcast;
    }
    else {
        // Each worker task will throw this many darts.
        const Hits numThrows = s.totalNumThrows_ / numTasks();

        // compute pi for this task
        const Hits hits = throwDarts(numThrows);

        std::cout << "Task " << taskId() << " had " << hits <<
            " hits out of " << numThrows << " throws" << std::endl;

        Hits sumHits = 0; // sum of ALL subprocess hits
        if (!MPIOK(MPI_Barrier(comm()))) {
            ret = ErrBarrier;
        }
        else if (!mpiReduceSumHits(hits, sumHits)) {
            ret = ErrReduce;
        }
    }
    return ret;
}


bool
MpiCalcPi::mpiReduceSumHits(const Hits &sendbuf, Hits &recvbuf, const int count,
    const int root)
{
    return mpiReduce(&sendbuf, &recvbuf, count, MPI_INT64_T, MPI_SUM,
        ((-1 == root) ? managerTaskId() : root));
}


MpiCalcPi::Hits
MpiCalcPi::throwDarts(const Hits numDarts) const
{
    std::hash<long long> hll;
    const std::size_t rngSeed{ hll(hll(taskId() + time(nullptr)) +
        hll(std::chrono::system_clock::now().time_since_epoch().count())) };
    // The random number generator
    std::mt19937_64 rng(rngSeed);
    constexpr auto rngSpan{ rng.max() - rng.min() };
    auto randCoordSquared = [&rng, rngSpan]()->double {
        // calc random coord [-1.0, 1.0]
        const double coord{ ((2.0 * (rng() - rng.min())) / rngSpan) - 1.0 };
        return coord * coord;
    };

    // throw darts at unit-circle dart board
    Hits hits = 0;
    for (Hits n = 0; n < numDarts; ++n) {
        // Is (x^2 + y^2) <= 1.0^2 ?
        if ((randCoordSquared() + randCoordSquared()) <= 1.0) {
            // dart landed in circle! Increment hits.
            ++hits;
        }
    }

    return hits;
}


int
MpiCalcPi::processArgs(const StringArray1 &args, Settings &s)
{
    int ret = ErrNone;
    StringArray1::const_iterator it = args.cbegin();
    for (; it != args.cend(); ++it) {
        const std::string &arg{ *it };
        if (("-t" == arg) || ("--throws" == arg)) {
            if (++it == args.cend()) {
                ret = ErrArgs;
                break;
            }
            std::stringstream ss(*it);
            ss >> s.totalNumThrows_;
            std::cout << ">> set totalNumThrows=" << s.totalNumThrows_ <<
                std::endl;
        }
    }
    return ret;
}
