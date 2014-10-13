#ifndef _BONSAI_CATALYST_DATA_
#define _BONSAI_CATALYST_DATA_

#pragma once

#include "RendererData.h"
#include "vtkSmartPointer.h"

class vtkBonsaiPipeline;
class vtkCPProcessor;
class vtkCPDataDescription;
class vtkPolyData;

class BonsaiCatalystData : public RendererData
{
  public:
    BonsaiCatalystData(const int rank, const int nrank, const MPI_Comm &comm);
   ~BonsaiCatalystData();

    // virtual methods
   virtual void coProcess(double time, unsigned int timeStep);

   vtkSmartPointer<vtkCPProcessor> coProcessor;
   vtkSmartPointer<vtkCPDataDescription> coProcessorData;
   // IsTimeDataSet is meant to be used to make sure that
   // needtocoprocessthistimestep() is called before
   // calling any of the other coprocessing functions.
   // It is reset to false after calling coprocess as well
   // as if coprocessing is not needed for this time/time step
   bool isTimeDataSet;
   vtkSmartPointer<vtkPolyData> particles;
   vtkSmartPointer<vtkBonsaiPipeline> cxxPipeline;
};

#endif
