// based on: https://computing.llnl.gov/tutorials/mpi/#Point_to_Point_Routines

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <sstream>
#include <random>
#include <vector>

#include "mpi.h"

using Hits = uint64_t;
static_assert(sizeof(Hits) == sizeof(unsigned long long), "Size mismatch");

using StringArray1 = std::vector<std::string>;


enum ErrorCodes {
    ErrNone = 0,
    ErrVersion,
    ErrInit,
    ErrCommSize,
    ErrCommRank,
    ErrReduce,
    ErrFinalize,
    ErrBarrier,
    ErrBcast,
    ErrArgs
};


struct Settings {
    Hits totalNumThrows_{ int(5e6) }; // TOTAL throws at dartboard
};


static int
processArgs(const StringArray1 &args, Settings &s)
{
    int ret = ErrNone;
    StringArray1::const_iterator it = args.cbegin();
    for (;  it != args.cend(); ++it) {
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


//****************************************************************************
//****************************************************************************
//****************************************************************************

class MpiProcess {
public:
    MpiProcess(const MPI_Comm Comm = MPI_COMM_WORLD, const int managerTaskId = 0) :
        comm_(Comm),
        managerTaskId_(managerTaskId)
    {
    }


    ~MpiProcess()
    {
    }


    int
    run(const int argc, char *argv[])
    {
        int ret = ErrNone;
        if (!MPIOK(MPI_Init(&argc, &argv))) {
            ret = ErrInit; // fail
        }
        else if (!MPIOK(MPI_Comm_size(comm_, &numTasks_))) {
            ret = ErrCommSize; // fail
        }
        else if (!MPIOK(MPI_Comm_rank(comm_, &taskId_))) {
            ret = ErrCommRank; // fail
        }
        else {
            std::cout << "MPI task " << getTaskName() << " started" <<
                std::endl;

            if (syncStarts_ && !MPIOK(MPI_Barrier(comm_))) {
                // Process start sync requested and failed
                ret = ErrBarrier;
            }
            else {
                StringArray1 args;
                args.insert(args.end(), argv + 1, argv + argc);

                if (managerTaskId_ == taskId_) {
                    ret = runAsManager(args);
                }
                else {
                    ret = runAsWorker(args);
                }

                if (ErrNone != ret) {
                    // ret already set - do not sync ends
                }
                else if (syncEnds_ && !MPIOK(MPI_Barrier(comm_))) {
                    // Process end sync requested and failed
                    ret = ErrBarrier;
                }
            }
        }

        // always call MPI_Finalize(). Don't change ret if error is already set.
        if (!MPIOK(MPI_Finalize()) && (ErrNone == ret)) {
            // All was okay until MPI_Finalize()
            ret = ErrFinalize;
        }

        std::cout << "MPI task " << getTaskName() << " ending" << std::endl;
        return ret;
    }


protected:

    bool
    mpiReduce(const void* sendbuf, void* recvbuf, const int count,
        const MPI_Datatype datatype, const MPI_Op op, const int root = -1)
    {
        return MPIOK(MPI_Reduce(sendbuf, recvbuf, count, datatype, op,
            ((-1 == root) ? managerTaskId_ : root), comm_));
    }


    bool
    mpiReduceSum(const Hits &sendbuf, Hits &recvbuf, const int count = 1,
        const int root = -1)
    {
        return mpiReduce(&sendbuf, &recvbuf, count, MPI_INT64_T, MPI_SUM,
            ((-1 == root) ? managerTaskId_ : root));
    }


    std::string &
    getTaskName()
    {
        if (taskName_.empty()) {
            std::stringstream ss;

            int len = -1;
            char commName[MPI_MAX_OBJECT_NAME]{ 0 };
            if (MPIOK(MPI_Comm_get_name(comm_, commName, &len))) {
                ss << commName;
            }
            else {
                ss << "NULL_COMMNAME";
            }

            ss << "." << taskId_ << "@";

            char procName[MPI_MAX_PROCESSOR_NAME]{ 0 };
            if (MPIOK(MPI_Get_processor_name(procName, &len))) {
                ss << procName;
            }
            else {
                ss << "NULL_PROCNAME";
            }

            taskName_ = ss.str();
        }
        return taskName_;
    }


    std::string &
    getVersionString() const
    {
        if (libVerStr_.empty()) {
            std::stringstream ss;

            char libVersionStr[MPI_MAX_LIBRARY_VERSION_STRING]{ '\0' };
            int len;
            if (MPIOK(MPI_Get_library_version(libVersionStr, &len))) {
                ss << libVersionStr;
            }
            else {
                ss << "NULL_LIB_VERSION";
            }

            ss << " API(";
            int version;
            int subversion;
            if (MPIOK(MPI_Get_version(&version, &subversion))) {
                ss << version << "." << subversion;
            }
            else {
                ss << "NULL";
            }
            ss << ")";
            libVerStr_ = ss.str();
        }
        return libVerStr_;
    }


    int         numTasks() const {
                    return numTasks_; }


    int         taskId() const {
                    return taskId_; }


    int         managerTaskId() const {
                    return managerTaskId_; }


    MPI_Comm    comm() const {
                    return comm_; }


    bool        MPIOK(const int rc) const {
                    return MPI_SUCCESS == (rc); }


private:
    int runAsManager(const StringArray1 &args)
    {
        return this->runAsManagerImpl(args);
    }

    virtual int runAsManagerImpl(const StringArray1 &args) = 0;

    int runAsWorker(const StringArray1 &args)
    {
        return this->runAsWorkerImpl(args);
    }

    virtual int runAsWorkerImpl(const StringArray1 &args) = 0;


private:
    bool                syncStarts_{ true };
    bool                syncEnds_{ false };
    mutable std::string libVerStr_;
    MPI_Comm            comm_;
    int                 numTasks_{ 0 }; // # tasks including managerTaskId_
    int                 taskId_{ -1 };
    mutable std::string taskName_;
    int                 managerTaskId_{ -1 };
};



//****************************************************************************
//****************************************************************************
//****************************************************************************

class MpiCalcPi : public MpiProcess {
public:
    MpiCalcPi() :
        MpiProcess()
    {
    }

    ~MpiCalcPi()
    {
    }

private:
    int
    runAsManagerImpl(const StringArray1 &args) override
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
            else if (!mpiReduceSum(hits, sumHits)) {
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
    runAsWorkerImpl(const StringArray1 &args) override
    {
        int ret = ErrNone;
        Settings s;
        if (!MPIOK(MPI_Bcast(&s, sizeof(s), MPI_UNSIGNED_CHAR, managerTaskId(), comm()))) {
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
            else if (!MPIOK(MPI_Reduce(&hits, &sumHits, 1, MPI_INT64_T, MPI_SUM,
                managerTaskId(), comm()))) {
                ret = ErrReduce;
            }
        }
        return ret;
    }


private:
    Hits
    throwDarts(const Hits numDarts)
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
};



int
main(int argc, char *argv[])
{
    MpiCalcPi p;
    return p.run(argc, argv);
}
