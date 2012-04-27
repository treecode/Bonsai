                     README for cuxTimer

Executive Summary
-----------------
cuxTimer is a simple code-timing profiling tool which works through code instrumentation. By inserting a single line of code at any point in a CUDA program, the subsequent section of code is timed on a per-thread basis. Average and maximum execution times across all threads are recorded, with results aggregated into a summary.

The cuxTimer tool can operate entirely from device-code without need for host-side setup or tear-down, although it will run more slowly if not initialised from the host. Initialisation, reset and counter-display functions are available from both host and device code.

Up to 1023 unique timers can be tracked. Use of cuxTimer will increase execution wall-clock time, but every effort has been made to ensure that the times reported do not include the execution of cuxTimer itself.

cuxTimer works on all architectures from SM-1.1 and up. Owing to lack of 64-bit atomic support, counters on SM 1.1 can easily overflow so care should be taken interpreting results of programs which run for more than several milliseconds. For SM-1.1 architectures only, cuxTimer allocates 128 bytes of shared memory.


Quick Start
-----------
To begin, include the following file at the top of your main program:
		#include "cuxTimer_host.cu"

In your host-side code
Add this line to your main() somewhere before the first kernel launch:
		cudaxTimerReset();

and add this line after you have called cudaThreadSynchronize():
		cudaxTimerDisplay();

In your device code
At start of any section of device code which you want timed, simply add the following line:
		CUXTIMER();			// Assigns timer name & ID automatically

You may specify a more human-readable timer name like this:
		CUXTIMER("name");		// Assigns timer ID automatically

You may also set a specific ID number for your timer:
		CUXTIMER("name", id);	// id is an integer from 0-1023

The timer follows normal C++ scoping rules: it is live between { } - that is, the timer will count up to the "}" at the end of the scope in which the counter was declared.

A timer will pause if a new timer is declared while it is still active - when the new timer goes out of scope, the old timer will resume (think of a stack of timers). You can add up to 1023 separate timers to your code. Use a different id for each timing point, unless you want different section times to add together.

You can control cuxTimer entirely from the device, including timer display - for full details, see the complete documentation at "//sw/gpgpu/doc/tools/cuxTimer.doc".

(Note: For gcc < 4.3, if you omit the ID you might run into trouble because we use the line number as an auto-ID, which is not guaranteed to be unique.)


What cuxTimer Actually Counts
-----------------------------
cuxTimer counts GPU clock cycles individually for each thread, from when the timing point is created until it goes out of scope. It then combines the times for all threads together to give an average value. Note that the exact runtime for any one thread may vary, because hardware scheduling might give it more or less time during any given run.

This means that you may see slightly different average times if you run your program, because thread scheduling will differ between runs. This will be most apparent with only a few warps active; with many threads, variations will average out. Run your program a few times to see what the variation looks like.

You will only get the exact time taken by a thread if you run one single warp on the GPU. 
More Information
Full cuxTimer documentation is available at:
      //sw/gpgpu/doc/tools/cuxTimer.doc
      
All cuxTimer files, along with an example program cuxTimer_example.cu, can be found at:
      //sw/gpgpu/cuda/platform/cuxTimer/release



Example
-------
Let's take a simple program which implements memset() and use it to illustrate use of different timers, scopes, etc. We'll set 1023 bytes, because it's a non-power-of-2. We'll do it two ways - byte by byte, then word-by-word, and use cuxTimer to compare the performance of each:

#include "cuxTimer_host.cu"

// This is the function we'll be timing
__device__ void memset(void *ptr, int value, int length)
{
	CUXTIMER("orig_memset");		// Slow memset is ID 1
	for(int i=0; i<length; i++)
		((char *)ptr)[i] = value;

	// Here, timer 2 will take over from timer 1, pausing timer 1
	CUXTIMER("faster_memset");		// Fast memset is ID 2
    	value &= 0xFF;
    	int val1 = (value << 24) | (value << 16) | (value << 8) | value;
	for(int i=0; i<(length>>2); i++)
		((int *)ptr)[i] = val1;
	for(int i=(length & ~3); i<length; i++)
		((char *)ptr)[i] = value;

}	// Both timers stop, because both go out of scope
__global__ void time_memset()
{
	char *ptr = (char *)malloc(1023);
	memset(ptr, 0, 1023);
       free(ptr);
}
int main()
{
	cudaxTimerReset();
	time_memset<<< 10, 1 >>>();
	cudaThreadSynchronize();
	cudaxTimerDisplay();
	return 0;
}


We get the following output:

Timing stats:
                Name:    Count   Tot Cycles   Max Cycles   Avg Cycles
       faster_memset:       10       101364        10290        10136
         orig_memset:       10      1071442       107436       107144
---------------------------------------------------------------------
               TOTAL:       20      1172806       117726       117280

We can see that the fast memset is 4x faster than the slow memset (as you'd expect). Note that "orig_memset" does not include the time for "faster_memset" because the orig_memset timer is halted by the creation of the faster_memset timer.

A more complete example is available at:
      //sw/gpgpu/cuda/platform/cuxTimer/release/cuxTimer_example.cu

