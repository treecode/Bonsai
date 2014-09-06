/*
 * %Id: xrandpfile.c,v 1.4 2003/10/18 21:43:05 gustav Exp %
 *
 * %Log: xrandpfile.c,v %
 * Revision 1.4  2003/10/18 21:43:05  gustav
 * integers -> bytes
 *
 * Revision 1.3  2003/10/18 21:33:35  gustav
 * Indented the program (used emacs).
 *
 * Revision 1.2  2003/10/18 20:43:24  gustav
 * Added reading of status with MPI_Get_count
 *
 * Revision 1.1  2003/10/18 19:52:44  gustav
 * Initial revision
 *
 *
 */

#include <stdio.h>   /* all IO stuff lives here */
#include <stdlib.h>  /* exit lives here */
#include <unistd.h>  /* getopt lives here */
#include <string.h>  /* strcpy lives here */
#include <limits.h>  /* INT_MAX lives here */
#include <mpi.h>     /* MPI and MPI-IO live here */

#define MASTER_RANK 0
#define TRUE 1
#define FALSE 0
#define BOOLEAN int
#define MBYTE 1048576
#define SYNOPSIS printf ("synopsis: %s -f <file>\n", argv[0])

int main(argc, argv)
     int argc;
     char *argv[];
{
  /* my variables */

  int my_rank, pool_size, last_guy, i, count;
  BOOLEAN i_am_the_master = FALSE, input_error = FALSE;
  char *filename = NULL, *read_buffer;
  int filename_length;
  int *junk;
  int file_open_error, number_of_bytes;

  /* MPI_Offset is long long */

  MPI_Offset my_offset, my_current_offset, total_number_of_bytes,
    number_of_bytes_ll, max_number_of_bytes_ll;
  MPI_File fh;
  MPI_Status status;
  double start, finish, io_time, longest_io_time;

  /* getopt variables */

  extern char *optarg;
  int c;

  /* ACTION */

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &pool_size);
  last_guy = pool_size - 1;
  if (my_rank == MASTER_RANK) i_am_the_master = TRUE;

  if (i_am_the_master) {

    /* read the command line */

    while ((c = getopt(argc, argv, "f:h")) != EOF)
      switch(c) {
      case 'f': 
	filename = optarg;
#ifdef DEBUG
	printf("input file: %s\n", filename);
#endif
	break;
      case 'h':
	SYNOPSIS;
	input_error = TRUE;
	break;
      case '?':
	SYNOPSIS;
	input_error = TRUE;
	break;
      } /* end of switch(c) */

    /* Check if the command line has initialized filename and
     * number_of_blocks.
     */

    if (filename == NULL) {
      SYNOPSIS;
      input_error = TRUE;
    }

    if (input_error) MPI_Abort(MPI_COMM_WORLD, 1);

    filename_length = strlen(filename) + 1;

    /* This is another way of exiting, but it can be done only
       if no files have been opened yet. */

  } /* end of "if (i_am_the_master)"; reading the command line */

    /* If we got this far, the data read from the command line
       should be OK. */
  
  MPI_Bcast(&filename_length, 1, MPI_INT, MASTER_RANK, MPI_COMM_WORLD);
  if (! i_am_the_master) filename = (char*) malloc(filename_length);
#ifdef DEBUG
  printf("%3d: allocated space for filename\n", my_rank);
#endif
  MPI_Bcast(filename, filename_length, MPI_CHAR, MASTER_RANK, MPI_COMM_WORLD);
#ifdef DEBUG
  printf("%3d: received broadcast\n", my_rank);
  printf("%3d: filename = %s\n", my_rank, filename);
#endif

  MPI_Barrier(MPI_COMM_WORLD);

  /* Default I/O error handling is MPI_ERRORS_RETURN */

  file_open_error = MPI_File_open(MPI_COMM_WORLD, filename, 
		                  MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

  if (file_open_error != MPI_SUCCESS) {

    char error_string[BUFSIZ];
    int length_of_error_string, error_class;

    MPI_Error_class(file_open_error, &error_class);
    MPI_Error_string(error_class, error_string, &length_of_error_string);
    printf("%3d: %s\n", my_rank, error_string);

    MPI_Error_string(file_open_error, error_string, &length_of_error_string);
    printf("%3d: %s\n", my_rank, error_string);

    MPI_Abort(MPI_COMM_WORLD, file_open_error);
  }

  MPI_File_get_size(fh, &total_number_of_bytes);
#ifdef DEBUG
  printf("%3d: total_number_of_bytes = %lld\n", my_rank, total_number_of_bytes);
#endif

  number_of_bytes_ll = total_number_of_bytes / pool_size;

  /* If pool_size does not divide total_number_of_bytes evenly,
     the last process will have to read more data, i.e., to the
     end of the file. */

  max_number_of_bytes_ll = 
    number_of_bytes_ll + total_number_of_bytes % pool_size;

  if (max_number_of_bytes_ll < INT_MAX) {

    if (my_rank == last_guy)
      number_of_bytes = (int) max_number_of_bytes_ll;
    else
      number_of_bytes = (int) number_of_bytes_ll;

    read_buffer = (char*) malloc(number_of_bytes);
#ifdef DEBUG
    printf("%3d: allocated %d bytes\n", my_rank, number_of_bytes);
#endif

    my_offset = (MPI_Offset) my_rank * number_of_bytes_ll;
#ifdef DEBUG
    printf("%3d: my offset = %lld\n", my_rank, my_offset);
#endif
    MPI_File_seek(fh, my_offset, MPI_SEEK_SET);

    MPI_Barrier(MPI_COMM_WORLD);

    start = MPI_Wtime();
    MPI_File_read(fh, read_buffer, number_of_bytes, MPI_BYTE, &status);
    finish = MPI_Wtime();
    MPI_Get_count(&status, MPI_BYTE, &count);
#ifdef DEBUG
    printf("%3d: read %d bytes\n", my_rank, count);
#endif
    MPI_File_get_position(fh, &my_offset);
#ifdef DEBUG
    printf("%3d: my offset = %lld\n", my_rank, my_offset);
#endif

    io_time = finish - start;
    MPI_Allreduce(&io_time, &longest_io_time, 1, MPI_DOUBLE, MPI_MAX,
		  MPI_COMM_WORLD);
    if (i_am_the_master) {
      printf("longest_io_time       = %f seconds\n", longest_io_time);
      printf("total_number_of_bytes = %lld\n", total_number_of_bytes);
      printf("transfer rate         = %f MB/s\n", 
	     total_number_of_bytes / longest_io_time / MBYTE);
    }
  }
  else {
    if (i_am_the_master) {
      printf("Not enough memory to read the file.\n");
      printf("Consider running on more nodes.\n");
    }
  } /* of if(max_number_of_bytes_ll < INT_MAX) */

  MPI_File_close(&fh);

  MPI_Finalize();
  exit(0);
}

