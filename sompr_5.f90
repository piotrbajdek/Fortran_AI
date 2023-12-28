! BSD 3-Clause No Military License
! Copyright © 2023, Piotr Bajdek. All Rights Reserved.

program sompr_5
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: qp = selected_real_kind(33, 4931)
  real(qp), dimension(:), allocatable :: sequence
  real(qp) :: b = 0.0_qp, alpha = 0.001_qp, beta1 = 0.9_qp, beta2 = 0.999_qp, t = 0.0_qp
  real(qp) :: mt(20) = 0.0_qp, vt(20) = 0.0_qp, m_hat(20), v_hat(20), denom(20)
  real(qp), dimension(20) :: w, dw
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
  w = 0.0_qp

  ! Optimization loop using Adam optimizer
  do i=1, iterations
      ! Initialize gradients
      dw = 0.0_qp

      ! Compute for each element of the sequence excluding the first five
      do j=6, seq_len
          y_pred = w(1) * sequence(j-2) &
          + w(2) * sequence(j-3) &
          + w(3) * sequence(j-4) &
          + w(4) * sequence(j-5) &
          + w(5) * sequence(j-6) &
          + w(6) * sequence(j-2)**2 &
          + w(7) * sequence(j-3)**2 &
          + w(8) * sequence(j-4)**2 &
          + w(9) * sequence(j-5)**2 &
          + w(10) * sequence(j-6)**2 &
          + w(11) * sequence(j-2) * sequence(j-3) &
          + w(12) * sequence(j-2) * sequence(j-4) &
          + w(13) * sequence(j-2) * sequence(j-5) &
          + w(14) * sequence(j-2) * sequence(j-6) &
          + w(15) * sequence(j-3) * sequence(j-4) &
          + w(16) * sequence(j-3) * sequence(j-5) &
          + w(17) * sequence(j-3) * sequence(j-6) &
          + w(18) * sequence(j-4) * sequence(j-5) &
          + w(19) * sequence(j-4) * sequence(j-6) &
          + w(20) * sequence(j-5) * sequence(j-6) &
          + b
          error = y_pred - sequence(j)
          dw(1) = dw(1) + error * sequence(j-2)
          dw(2) = dw(2) + error * sequence(j-3)
          dw(3) = dw(3) + error * sequence(j-4)
          dw(4) = dw(4) + error * sequence(j-5)
          dw(5) = dw(5) + error * sequence(j-6)
          dw(6) = dw(6) + error * sequence(j-2)**2
          dw(7) = dw(7) + error * sequence(j-3)**2
          dw(8) = dw(8) + error * sequence(j-4)**2
          dw(9) = dw(9) + error * sequence(j-5)**2
          dw(10) = dw(10) + error * sequence(j-6)**2
          dw(11) = dw(11) + error * sequence(j-2) * sequence(j-3)
          dw(12) = dw(12) + error * sequence(j-2) * sequence(j-4)
          dw(13) = dw(13) + error * sequence(j-2) * sequence(j-5)
          dw(14) = dw(14) + error * sequence(j-2) * sequence(j-6)
          dw(15) = dw(15) + error * sequence(j-3) * sequence(j-4)
          dw(16) = dw(16) + error * sequence(j-3) * sequence(j-5)
          dw(17) = dw(17) + error * sequence(j-3) * sequence(j-6)
          dw(18) = dw(18) + error * sequence(j-4) * sequence(j-5)
          dw(19) = dw(19) + error * sequence(j-4) * sequence(j-6)
          dw(20) = dw(20) + error * sequence(j-5) * sequence(j-6)
          db = db + error
      end do

      ! Normalize gradients
      dw = dw / (seq_len - 5)
      db = db / (seq_len - 5)

      ! Update weights using the Adam optimization algorithm
      t = t + 1.0_qp
      mt = beta1 * mt + (1.0_qp - beta1) * dw
      vt = beta2 * vt + (1.0_qp - beta2) * (dw**2)
      m_hat = mt / (1.0_qp - beta1**t)
      v_hat = vt / (1.0_qp - beta2**t)
      denom = sqrt(v_hat) + epsilon
      w = w - alpha * m_hat / denom
      b = b - alpha * db

  end do

  ! Predict the value based on the last five elements of the sequence
  ! Second Order Multivariate Polynomial Regression
  y_pred = w(1) * sequence(seq_len-2) &
      + w(2) * sequence(seq_len-3) &
      + w(3) * sequence(seq_len-4) &
      + w(4) * sequence(seq_len-5) &
      + w(5) * sequence(seq_len-6) &
      + w(6) * sequence(seq_len-2)**2 &
      + w(7) * sequence(seq_len-3)**2 &
      + w(8) * sequence(seq_len-4)**2 &
      + w(9) * sequence(seq_len-5)**2 &
      + w(10) * sequence(seq_len-6)**2 &
      + w(11) * sequence(seq_len-2) * sequence(seq_len-3) &
      + w(12) * sequence(seq_len-2) * sequence(seq_len-4) &
      + w(13) * sequence(seq_len-2) * sequence(seq_len-5) &
      + w(14) * sequence(seq_len-2) * sequence(seq_len-6) &
      + w(15) * sequence(seq_len-3) * sequence(seq_len-4) &
      + w(16) * sequence(seq_len-3) * sequence(seq_len-5) &
      + w(17) * sequence(seq_len-3) * sequence(seq_len-6) &
      + w(18) * sequence(seq_len-4) * sequence(seq_len-5) &
      + w(19) * sequence(seq_len-4) * sequence(seq_len-6) &
      + w(20) * sequence(seq_len-5) * sequence(seq_len-6) &
      + b

  ! Display the prediction result
  write(*,'(a,F0.33)') 'Predicted value = ', y_pred

  ! Deallocate memory
  deallocate(sequence)

end program sompr_5
