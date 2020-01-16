// based on: https://computing.llnl.gov/tutorials/mpi/#Point_to_Point_Routines

#include <chrono>
#include <functional>
#include <iostream>
#include <string>
#include <random>

#include "mpi.h"

using Hits = uint64_t;

static constexpr bool
MPIOK(const int rc) {
    return MPI_SUCCESS == (rc);
}


static Hits
dboardHits(const Hits numDarts, const int taskId)
{
    std::hash<long long> hll;
    const std::size_t rngSeed{ hll(hll(taskId + time(nullptr)) +
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


static const char *
getTaskName(const MPI_Comm comm, const int taskId)
{
    static char name[MPI_MAX_PROCESSOR_NAME + MPI_MAX_OBJECT_NAME + 16]{ 0 };
    if ('\0' == name[0]) {
        char procName[MPI_MAX_PROCESSOR_NAME]{ 0 };
        int len = -1;
        if (('\0' == procName[0]) && !MPIOK(MPI_Get_processor_name(procName, &len))) {
            strcpy_s(procName, "??");
        }

        char commName[MPI_MAX_OBJECT_NAME]{ 0 };
        if (!MPIOK(MPI_Comm_get_name(comm, commName, &len))) {
            strcpy_s(commName, "??");
        }

        sprintf_s(name, "%s.%d@%s", commName, taskId, procName);
    }
    return name;
}


enum ErrorCodes {
    ErrNone = 0,
    ErrVersion = -1,
    ErrInit = -2,
    ErrCommSize = -3,
    ErrCommRank = -4,
    ErrReduce = -5,
    ErrFinalize = -6,
};

static_assert(sizeof(Hits) == sizeof(unsigned long long), "Size mismatch");

int
main(int argc, char *argv[])
{
    constexpr int MASTER{ 0 }; // master task ID
    constexpr MPI_Comm Comm{ MPI_COMM_WORLD };

    int ret = ErrNone;
    int taskId = 0;	// task ID - also used as seed number
    int numtasks = 0;   // # tasks inluding MASTER

    // Obtain number of tasks and task ID
    if (!MPIOK(MPI_Init(&argc, &argv))) {
        ret = ErrInit; // fail
    }
    else if (!MPIOK(MPI_Comm_size(Comm, &numtasks))) {
        ret = ErrCommSize; // fail
    }
    else if (!MPIOK(MPI_Comm_rank(Comm, &taskId))) {
        ret = ErrCommRank; // fail
    }
    else {
        const bool isMasterTask = (MASTER == taskId);

        Hits totalNumThrows{ int(5e6) }; // TOTAL throws at dartboard

        int ii = 1;
        while (ii < argc) {
            const std::string arg{ argv[ii++] };
            if ((("-t" == arg) || ("--throws" == arg)) && (ii < argc)) {
                totalNumThrows = atoll(argv[ii++]);
                if (isMasterTask) {
                    std::cout << ">> set totalNumThrows=" << totalNumThrows << std::endl;
                }
            }
        }

        int version;
        int subversion;
        if (isMasterTask && MPIOK(MPI_Get_version(&version, &subversion))) {
            printf("MPI version %d.%d\n", version, subversion);
            fflush(stdout);
        }

        printf("MPI task %s has started...\n", getTaskName(Comm, taskId));
        fflush(stdout);

        // Each subtask will throw this many darts
        Hits numThrows = totalNumThrows / numtasks;
        if (isMasterTask) {
            // Master task will pick up any throws lost to integer truncation
            numThrows += (totalNumThrows % numThrows);
        }

        // Use MPI_Reduce to sum values of piTask across all tasks and stores
        // the accumulated value in sumHits:
        //  piTask     - The send buffer
        //  sumHits      - The receive buffer (used by the receiving task only)
        //  1          - Number of values pointed to by &piTask
        //  MPI_DOUBLE - Type of sumHits
        //  MPI_SUM    - A pre-defined, MPI_Op handle (double-precision floating
        //               point vector addition). Must be declared extern.
        //  MASTER is the task that will receive the result of the reduction
        //  operation
        //  Comm is the group of tasks that will participate.

        // compute pi for this task
        Hits hits = dboardHits(numThrows, taskId);

        // This task output will likely be scrambled/interlaced with the output
        // of the other tasks in Comm.
        printf("Task %d had %llu hits out of %llu throws\n", taskId, hits,
            numThrows);
        fflush(stdout);

        Hits sumHits = 0; // sum of ALL subprocess hits
        if (!MPIOK(MPI_Reduce(&hits, &sumHits, 1, MPI_INT64_T, MPI_SUM, MASTER,
                Comm))) {
            ret = ErrReduce;
        }
        else if (isMasterTask) {
            // Master and all subtasks have computed their values for PI. The
            // call to MPI_Reduce() has summed them all together and placed
            // result into sumHits.
            printf("After %llu throws...\n", totalNumThrows);
            fflush(stdout);
            const double computedPi{ (4.0 * sumHits) / totalNumThrows };
            const double actualPi{ 3.1415926535897 };
            printf("  Computed PI: %.8f\n", computedPi);
            fflush(stdout);
            printf("  Actual   PI: %.8f\n", actualPi);
            fflush(stdout);
            printf("  Error      : %g\n", actualPi - computedPi);
            fflush(stdout);
        }
    }
    // always call MPI_Finalize(). Don't change ret if error is already set.
    if (!MPIOK(MPI_Finalize()) && (ErrNone == ret)) {
        // All was okay until MPI_Finalize()
        ret = ErrFinalize;
    }

    printf("MPI task %s ending\n", getTaskName(Comm, taskId));
    fflush(stdout);
    return ret;
}
