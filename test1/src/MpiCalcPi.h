#ifndef MPICALCPI_H
#define MPICALCPI_H

#include "MpiProcess.h"

struct Settings;


//****************************************************************************
//****************************************************************************
//****************************************************************************

class MpiCalcPi : public MpiProcess {
public:
    using Hits = uint64_t;
    static_assert(sizeof(Hits) == sizeof(unsigned long long), "Size mismatch");

public:
    MpiCalcPi();

    ~MpiCalcPi();

private:
    int         runAsManagerImpl(const StringArray1 &args) override;

    int         runAsWorkerImpl(const StringArray1 &args) override;

    bool        mpiReduceSumHits(const Hits &sendbuf, Hits &recvbuf,
                    const int count = 1, const int root = -1);


    Hits        throwDarts(const Hits numDarts) const;


    static int  processArgs(const StringArray1 &args, Settings &s);
};

#endif // MPICALCPI_H
