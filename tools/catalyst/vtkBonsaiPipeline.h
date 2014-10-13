#ifndef VTKCPPVSMPIPELINE_H
#define VTKCPPVSMPIPELINE_H

#include <vtkCPPipeline.h>
#include <string>

class vtkCPDataDescription;
class vtkCPPythonHelper;

class vtkBonsaiPipeline : public vtkCPPipeline
{
public:
  static vtkBonsaiPipeline* New();
  vtkTypeMacro(vtkBonsaiPipeline,vtkCPPipeline);
  virtual void PrintSelf(ostream& os, vtkIndent indent);

  virtual void Initialize(int outputFrequency, std::string& fileName);

  virtual int RequestDataDescription(vtkCPDataDescription* dataDescription);

  virtual int CoProcess(vtkCPDataDescription* dataDescription);

protected:
  vtkBonsaiPipeline();
  virtual ~vtkBonsaiPipeline();

private:
  vtkBonsaiPipeline(const vtkBonsaiPipeline&); // Not implemented
  void operator=(const vtkBonsaiPipeline&); // Not implemented

  int OutputFrequency;
  std::string FileName;
};
#endif
