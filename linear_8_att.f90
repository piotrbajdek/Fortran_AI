! BSD 3-Clause No Military License
! Copyright © 2023, Piotr Bajdek. All Rights Reserved.

program linear_8_att
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: qp = selected_real_kind(33, 4931)
  real(qp), dimension(:,:), allocatable :: attention_weights
  real(qp), dimension(:), allocatable :: sequence
  real(qp), dimension(8) :: w, dw
  real(qp) :: b = 0.0_qp, alpha = 0.001_qp, beta1 = 0.9_qp, beta2 = 0.999_qp, t = 0.0_qp
  real(qp) :: mt(8) = 0.0_qp, vt(8) = 0.0_qp, m_hat(8), v_hat(8), denom(8)
  real(qp) :: db, y_pred, error
  real(qp) :: epsilon = 1.0E-8_qp
  integer :: i, j, iterations = 12500, seq_len

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
  w = [0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp]

  ! Initialize attention weights
  allocate(attention_weights(seq_len, seq_len))
  attention_weights = 0.0_qp

  ! Optimization loop using Adam optimizer
  do i=1, iterations
      ! Initialize gradients
      dw = [0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp]

      ! Element-wise dot-product attention calculation
      do j = 1, seq_len
        attention_weights(:, j) = sequence(j) * sequence(:)
      end do

      ! Scale attention weights
      attention_weights = attention_weights / real(seq_len)

      ! Compute gradients using attention weights
      do j=9, seq_len
          y_pred = w(1) * sequence(j-2) &
          + w(2) * sequence(j-3) &
          + w(3) * sequence(j-4) &
          + w(4) * sequence(j-5) &
          + w(5) * sequence(j-6) &
          + w(6) * sequence(j-7) &
          + w(7) * sequence(j-8) &
          + w(8) * sequence(j-9) &
          + b
          error = y_pred - sequence(j)
          dw(1) = dw(1) + error * sequence(j-2) * attention_weights(j-2, j)
          dw(2) = dw(2) + error * sequence(j-3) * attention_weights(j-3, j)
          dw(3) = dw(3) + error * sequence(j-4) * attention_weights(j-4, j)
          dw(4) = dw(4) + error * sequence(j-5) * attention_weights(j-5, j)
          dw(5) = dw(5) + error * sequence(j-6) * attention_weights(j-6, j)
          dw(6) = dw(6) + error * sequence(j-7) * attention_weights(j-7, j)
          dw(7) = dw(7) + error * sequence(j-8) * attention_weights(j-8, j)
          dw(8) = dw(8) + error * sequence(j-9) * attention_weights(j-9, j)
          db = db + error
      end do

      ! Normalize gradients
      dw = dw / (seq_len - 8)
      db = db / (seq_len - 8)

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

  ! Predict the value based on the last eight elements of the sequence
  y_pred = w(1) * sequence(seq_len-2) &
      + w(2) * sequence(seq_len-3) &
      + w(3) * sequence(seq_len-4) &
      + w(4) * sequence(seq_len-5) &
      + w(5) * sequence(seq_len-6) &
      + w(6) * sequence(seq_len-7) &
      + w(7) * sequence(seq_len-8) &
      + w(8) * sequence(seq_len-9) &
      + b

  ! Display the prediction result
  write(*,'(a,F0.33)') 'Predicted value = ', y_pred

  ! Deallocate memory
  deallocate(sequence, attention_weights)

end program linear_8_att
