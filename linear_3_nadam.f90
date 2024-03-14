! BSD 3-Clause No Military License
! Copyright © 2024, Piotr Bajdek. All Rights Reserved.

! Nesterov-accelerated Adaptive Moment Estimation
program linear_3_nadam
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: qp = selected_real_kind(33, 4931)
  real(qp), dimension(:), allocatable :: sequence
  real(qp) :: b = 0.0_qp, alpha = 0.001_qp, beta1 = 0.9_qp, beta2 = 0.999_qp, t = 0.0_qp
  real(qp) :: mt(3) = 0.0_qp, vt(3) = 0.0_qp, m_hat(3), mt_hat(3), v_hat(3), denom(3)
  real(qp) :: mt_prev(3) = 0.0_qp
  real(qp), dimension(3) :: w, dw
  real(qp) :: db, y_pred, error
  real(qp) :: epsilon = 1.0E-8_qp
  integer :: i, j, iterations = 100000, seq_len

  ! Check if an input file is provided
  if (command_argument_count() == 0) then
      write(*,'(a)') 'No input file selected!'
      call exit(0)
  end if

  ! Retrieve the input file name from command line arguments
  call get_command_argument(1, arg_1)

  ! Open the file, read the sequence length, and allocate memory
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

  ! Initialize the weight vector
  w = [0.0_qp, 0.0_qp, 0.0_qp]

  ! Optimization loop using Nadam optimizer
  do i=1, iterations
      ! Initialize gradients
      dw = [0.0_qp, 0.0_qp, 0.0_qp]

      ! Compute for each element of the sequence excluding the first three
      do j=4, seq_len
          y_pred = w(1) * sequence(j-2) &
          + w(2) * sequence(j-3) &
          + w(3) * sequence(j-4) &
          + b
          error = y_pred - sequence(j)
          dw(1) = dw(1) + error * sequence(j-2)
          dw(2) = dw(2) + error * sequence(j-3)
          dw(3) = dw(3) + error * sequence(j-4)
          db = db + error
      end do

      ! Normalize gradients
      dw = dw / (seq_len - 3)
      db = db / (seq_len - 3)

      ! Update weights using the Nadam optimization algorithm
      t = t + 1.0_qp
      mt = beta1 * mt + (1.0_qp - beta1) * dw
      mt_hat = mt / (1.0_qp - beta1**t)
      vt = beta2 * vt + (1.0_qp - beta2) * (dw**2)
      v_hat = vt / (1.0_qp - beta2**t)
      denom = sqrt(v_hat) + epsilon
      mt_prev = mt_prev * beta1 + (1.0_qp - beta1) * dw
      w = w - alpha * (mt_hat + beta1 * mt_prev / (1.0_qp - beta1**t))/ denom
      b = b - alpha * db

  end do

  ! Predict the value based on the last three elements of the sequence
  y_pred = w(1) * sequence(seq_len-2) &
      + w(2) * sequence(seq_len-3) &
      + w(3) * sequence(seq_len-4) &
      + b

  ! Display the prediction result
  write(*,'(a,F0.33)') 'Predicted value = ', y_pred

  ! Deallocate memory
  deallocate(sequence)

end program linear_3_nadam
