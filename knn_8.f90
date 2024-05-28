! BSD 3-Clause No Military License
! Copyright © 2024, Piotr Bajdek. All Rights Reserved.

program knn_8
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: qp = selected_real_kind(33, 4931)
  real(qp), dimension(:), allocatable :: sequence
  integer :: k = 8
  integer :: seq_len, i, j, m
  real(qp), dimension(8) :: x_new
  real(qp) :: y_pred, dist, min_dist
  real(qp), dimension(:,:), allocatable :: training_data
  real(qp), dimension(:), allocatable :: distances

  if (command_argument_count() == 0) then
      write(*,'(a)') 'No input file selected!'
      call exit(0)
  end if

  call get_command_argument(1, arg_1)

  open(10, file=arg_1, status='old', action='read')
  seq_len = 0
  do
      read(10,*,iostat=i)
      if (i /= 0) exit
      seq_len = seq_len + 1
  end do
  allocate(sequence(seq_len))
  rewind(10)
  do i=1, seq_len
      read(10,*) sequence(i)
  end do
  close(10)

  allocate(training_data(seq_len-8, 9))
  do i=1, seq_len-8
      training_data(i,1:8) = sequence(i:i+7)
      training_data(i,9) = sequence(i+8)
  end do

  x_new = sequence(seq_len-7:seq_len)

  allocate(distances(seq_len-8))

  do i=1, seq_len-8
      distances(i) = sqrt(sum((training_data(i,1:8) - x_new)**2))
  end do

  y_pred = 0.0_qp
  do i=1, k
      min_dist = minval(distances)
      do j=1, seq_len-8
          if (distances(j) == min_dist) then
              y_pred = y_pred + training_data(j, 9)
              distances(j) = huge(1.0_qp)
              exit
          end if
      end do
  end do

  y_pred = y_pred / k

  write(*,'(a,F0.33)') 'Predicted value = ', y_pred

  deallocate(sequence, training_data, distances)

end program knn_8
