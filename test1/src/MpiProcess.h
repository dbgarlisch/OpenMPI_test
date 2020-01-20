#ifndef MPIPROCESS_H
#define MPIPROCESS_H

#include <iostream>
#include <string>
#include <sstream>
#include <vector>

#include "mpi.h"


//****************************************************************************
//****************************************************************************
//****************************************************************************

class MpiProcess {
public:
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

    static const int    RootUseManager{ -1 };

    using StringArray1 = std::vector<std::string>;

public:
    MpiProcess(const MPI_Comm Comm = MPI_COMM_WORLD,
        const int managerTaskId = 0);


    ~MpiProcess();


    int             run(const int argc, char *argv[]);


protected:

    bool            mpiReduce(const void* sendbuf, void* recvbuf,
                        const int count, const MPI_Datatype datatype,
                        const MPI_Op op, const int root = RootUseManager);

    bool            mpiBcast(void* buf, const int count,
                        const MPI_Datatype datatype = MPI_UNSIGNED_CHAR,
                        const int root = RootUseManager);

    std::string &   getTaskName() const;

    std::string &   getVersionString() const;

    int             numTasks() const {
                        return numTasks_; }

    int             taskId() const {
                        return taskId_; }

    int             managerTaskId() const {
                        return managerTaskId_; }

    MPI_Comm        comm() const {
                        return comm_; }

    bool            MPIOK(const int rc) const {
                        return MPI_SUCCESS == (rc); }

private:
    int             runAsManager(const StringArray1 &args);

    int             runAsWorker(const StringArray1 &args);

    virtual int     runAsManagerImpl(const StringArray1 &args) = 0;

    virtual int     runAsWorkerImpl(const StringArray1 &args) = 0;


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

#endif // MPIPROCESS_H
