// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the DSREMOTECONNECT_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// DSREMOTECONNECT_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef DSREMOTECONNECT_EXPORTS
#define DSREMOTECONNECT_API __declspec(dllexport)
#else
#define DSREMOTECONNECT_API __declspec(dllimport)
#endif

#include "DSXInstance.h"
#include "DSXDCOMInstance.h"
#include "DSXNETInstance.h"
#include "DSXNETInstance.h"

typedef DSXInstance* DSXInstanceHandle;
typedef ChannelInstance* ChannelInstanceHandle;

void setConnType(int type);
int getConnType();
int initialize(const char* ip, const char* port);
int showSettings(int ch);
int listChannels(char* channelList);
int startMeasurement(int channelID, const char* port);
int stopMeasurement();
int resetMeasurement();
int getData(float* data, int N);
void hideDSX();
void showDSX();

extern "C" DSREMOTECONNECT_API int dsconCreateInstance(DSXInstanceHandle* handle, int connectionType);
extern "C" DSREMOTECONNECT_API int dsconConnect(DSXInstanceHandle handle, char* connectionString);
extern "C" DSREMOTECONNECT_API int dsconDestroyInstance(DSXInstanceHandle handle);
extern "C" DSREMOTECONNECT_API int dsconDisconnect(DSXInstanceHandle handle);
extern "C" DSREMOTECONNECT_API int dsconGetChannelCount(DSXInstanceHandle handle, size_t* count);
extern "C" DSREMOTECONNECT_API int dsconChannelGetName(ChannelInstanceHandle channel, char* name, size_t len);
extern "C" DSREMOTECONNECT_API int dsconSetAppWindowVisible(DSXInstanceHandle handle, bool visible);
extern "C" DSREMOTECONNECT_API int dsconChannelSetTransferred(ChannelInstanceHandle handle, bool transferred);
extern "C" DSREMOTECONNECT_API int dsconLoadSetup(DSXInstanceHandle handle, char* fileName);
extern "C" DSREMOTECONNECT_API int dsconIsInAqusition(DSXInstanceHandle handle, bool* value);
extern "C" DSREMOTECONNECT_API int dsconStartMeasurement(DSXInstanceHandle handle);
extern "C" DSREMOTECONNECT_API int dsconStopMeasurement(DSXInstanceHandle handle);
extern "C" DSREMOTECONNECT_API int dsconBeginRead(DSXInstanceHandle handle, double* startTime, double* endTime);
extern "C" DSREMOTECONNECT_API int dsconChannelReadScalarData(ChannelInstanceHandle handle, double** data, double** timestamps, size_t* count);
extern "C" DSREMOTECONNECT_API int dsconChannelReadScalarData_2(ChannelInstanceHandle handle, double* data, double* timestamps, size_t * count);
extern "C" DSREMOTECONNECT_API int dsconControlChannelWriteData(ChannelInstanceHandle handle, double data);

extern "C" DSREMOTECONNECT_API int dsconGetSampleRate(ChannelInstanceHandle handle, double* sampleRate);
extern "C" DSREMOTECONNECT_API int dsconGetChannelType(ChannelInstanceHandle handle, int* chType);
extern "C" DSREMOTECONNECT_API int dsconIsChannelControl(ChannelInstanceHandle handle, bool* result);
extern "C" DSREMOTECONNECT_API int dsconGetChUnit(ChannelInstanceHandle handle, char* resultUnit, size_t len);

extern "C" DSREMOTECONNECT_API int dsconEnumerateChannels(DSXInstanceHandle handle, ChannelIdList* channelIdList, size_t* count);
extern "C" DSREMOTECONNECT_API int dsconFreeEnumerateChannels(DSXInstanceHandle handle, ChannelIdList* channelIdList);
extern "C" DSREMOTECONNECT_API int dsconCreateChannelInstance(DSXInstanceHandle handle, ChannelID id, ChannelInstanceHandle* chInstance);
extern "C" DSREMOTECONNECT_API int dsconCreateChannelInstanceByName(DSXInstanceHandle handle, char* chName, ChannelInstanceHandle* chInstance);
extern "C" DSREMOTECONNECT_API int dsconFreeChannelInstance(ChannelInstanceHandle chInstance);

extern "C" DSREMOTECONNECT_API int dsconInitChannelIdEnum(DSXInstanceHandle handle);
extern "C" DSREMOTECONNECT_API int dsconGetNextChannelId(DSXInstanceHandle handle, ChannelID id);
extern "C" DSREMOTECONNECT_API int dsconGetNextChannelIdSize(DSXInstanceHandle handle, int* size);