! BSD 3-Clause No Military License
! Copyright © 2024, Piotr Bajdek. All Rights Reserved.

program linear_8_mh_att
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), dimension(:,:), allocatable :: attention_weights, key, query
  real(dp), dimension(:,:), allocatable :: value
  real(dp), dimension(:), allocatable :: sequence
  real(dp), dimension(8) :: w, dw
  real(dp) :: b = 0.0_dp, alpha = 0.001_dp, beta1 = 0.9_dp, beta2 = 0.999_dp, t = 0.0_dp
  real(dp) :: mt(8) = 0.0_dp, vt(8) = 0.0_dp, m_hat(8), v_hat(8), denom(8)
  real(dp) :: db, y_pred, error, scale_factor
  real(dp) :: epsilon = 1.0E-8_dp
  integer :: i, j, k, iterations = 12500, seq_len, head_size = 8

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
  w = 0.0_dp

  ! Initialize attention weights
  allocate(attention_weights(seq_len, seq_len), key(seq_len, head_size), query(seq_len, head_size), value(seq_len, head_size))
  attention_weights = 0.0_dp
  key = 0.0_dp
  query = 0.0_dp
  value = spread(sequence, dim=2, ncopies=head_size)

  ! Computing the query and key matrices
  do i = 1, seq_len
    do j = 1, head_size
      query(i, j) = sequence(i) * (j - 1)
      key(i, j) = sequence(i) * (j + 1)
    end do
  end do

  ! Optimization loop using Adam optimizer
  do i=1, iterations
      dw = 0.0_dp

      ! Multi-head attention calculation
      do j = 1, seq_len
        do k = 1, seq_len
          attention_weights(j, k) = sum(query(j, :) * key(k, :))
        end do
      end do

      ! Scale attention weights
      scale_factor = 1.0_dp / sqrt(real(head_size, dp))
      attention_weights = scale_factor * attention_weights

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
          dw(1) = dw(1) + error * sequence(j-2) * sum(attention_weights(j-2, :) * value(j, :))
          dw(2) = dw(2) + error * sequence(j-3) * sum(attention_weights(j-3, :) * value(j, :))
          dw(3) = dw(3) + error * sequence(j-4) * sum(attention_weights(j-4, :) * value(j, :))
          dw(4) = dw(4) + error * sequence(j-5) * sum(attention_weights(j-5, :) * value(j, :))
          dw(5) = dw(5) + error * sequence(j-6) * sum(attention_weights(j-6, :) * value(j, :))
          dw(6) = dw(6) + error * sequence(j-7) * sum(attention_weights(j-7, :) * value(j, :))
          dw(7) = dw(7) + error * sequence(j-8) * sum(attention_weights(j-8, :) * value(j, :))
          dw(8) = dw(8) + error * sequence(j-9) * sum(attention_weights(j-9, :) * value(j, :))
          db = db + error
      end do

      ! Normalize gradients
      dw = dw / (seq_len - 8)
      db = db / (seq_len - 8)

      ! Update weights using the Adam optimization algorithm
      t = t + 1.0_dp
      mt = beta1 * mt + (1.0_dp - beta1) * dw
      vt = beta2 * vt + (1.0_dp - beta2) * (dw**2)
      m_hat = mt / (1.0_dp - beta1**t)
      v_hat = vt / (1.0_dp - beta2**t)
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
  write(*,'(a,F0.13)') 'Predicted value = ', y_pred

  ! Deallocate memory
  deallocate(sequence, attention_weights, key, query, value)

end program linear_8_mh_att
