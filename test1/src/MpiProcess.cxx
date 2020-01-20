#include <iostream>
#include <string>
#include <sstream>

#include "MpiProcess.h"


MpiProcess::MpiProcess(const MPI_Comm Comm, const int managerTaskId) :
    comm_(Comm),
    managerTaskId_(managerTaskId)
{
}


MpiProcess::~MpiProcess()
{
}


int
MpiProcess::run(const int argc, char *argv[])
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


bool
MpiProcess::mpiReduce(const void* sendbuf, void* recvbuf, const int count,
    const MPI_Datatype datatype, const MPI_Op op, const int root)
{
    return MPIOK(MPI_Reduce(sendbuf, recvbuf, count, datatype, op,
        ((RootUseManager == root) ? managerTaskId_ : root), comm_));
}


bool
MpiProcess::mpiBcast(void* buf, const int count,
    const MPI_Datatype datatype, const int root)
{
    return MPIOK(MPI_Bcast(buf, count, datatype,
        ((RootUseManager == root) ? managerTaskId_ : root), comm_));
}


std::string &
MpiProcess::getTaskName() const
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
MpiProcess::getVersionString() const
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


int
MpiProcess::runAsManager(const StringArray1 &args)
{
    return this->runAsManagerImpl(args);
}


int
MpiProcess::runAsWorker(const StringArray1 &args)
{
    return this->runAsWorkerImpl(args);
}
