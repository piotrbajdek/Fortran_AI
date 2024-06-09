! BSD 3-Clause No Military License
! Copyright © 2024, Piotr Bajdek. All Rights Reserved.

program rnn_mh_att_3
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: dp = selected_real_kind(15, 307)
  real(dp), dimension(:), allocatable :: sequence
  real(dp), dimension(:,:,:), allocatable :: attention_weights
  real(dp) :: b = 0.0_dp, alpha = 0.001_dp, beta1 = 0.9_dp, beta2 = 0.999_dp, t = 0.0_dp
  real(dp) :: mt(9) = 0.0_dp, vt(9) = 0.0_dp, m_hat(9), v_hat(9), denom(9)
  real(dp), dimension(3) :: dw, db, dh, w_hy
  real(dp) :: y_pred, error, epsilon = 1.0E-8_dp, by, bh
  real(dp), dimension(3) :: h
  real(dp), dimension(3,3) :: w_xh, w_hh
  integer :: i, j, k, head, iterations = 100000, seq_len, num_heads = 2

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
  allocate(sequence(seq_len), attention_weights(seq_len, seq_len, num_heads))
  rewind(10)
  do i=1, seq_len
      read(10,*) sequence(i)
  end do
  close(10)

  ! Initialize the weight vector and biases
  w_xh = reshape([0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp], [3,3])
  w_hh = reshape([0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp, 0.0_dp], [3,3])
  w_hy = [0.0_dp, 0.0_dp, 0.0_dp]
  bh = 0.0_dp
  by = 0.0_dp
  h = [0.0_dp, 0.0_dp, 0.0_dp]

  ! Initialize attention weights
  attention_weights = 0.0_dp

  ! Optimization loop using Adam optimizer
  do i=1, iterations
      ! Initialize gradients
      dw = [0.0_dp, 0.0_dp, 0.0_dp]
      dh = [0.0_dp, 0.0_dp, 0.0_dp]
      db = [0.0_dp, 0.0_dp, 0.0_dp]

      ! Multi-head attention calculation
      do head = 1, num_heads
          do j = 1, seq_len
              attention_weights(:, j, head) = sequence(j) * sequence(:)
          end do

          ! Scale attention weights
          attention_weights(:, :, head) = attention_weights(:, :, head) / real(seq_len)
      end do

      ! Compute for each element of the sequence excluding the first three
      do j=4, seq_len
          ! Initialize combined hidden state
          h = [0.0_dp, 0.0_dp, 0.0_dp]

          ! Compute hidden state for each head
          do head = 1, num_heads
              h = h + tanh(matmul(w_xh, [sequence(j-1), sequence(j-2), sequence(j-3)]) + matmul(w_hh, h) + bh) * attention_weights(j-2:j, j, head)
          end do

          ! Average the contributions from all heads
          h = h / real(num_heads)

          ! Compute output with combined attention
          y_pred = dot_product(w_hy, h) + by

          error = y_pred - sequence(j)
          dw = dw + error * [sequence(j-1), sequence(j-2), sequence(j-3)]
          dh = dh + error * h
          db = db + error
      end do

      ! Normalize gradients
      dw = dw / (seq_len - 3)
      dh = dh / (seq_len - 3)
      db = db / (seq_len - 3)

      ! Update weights using the Adam optimization algorithm
      t = t + 1.0_dp
      do k = 1, 3
          mt(k) = beta1 * mt(k) + (1.0_dp - beta1) * dw(k)
          vt(k) = beta2 * vt(k) + (1.0_dp - beta2) * (dw(k)**2)
          m_hat(k) = mt(k) / (1.0_dp - beta1**t)
          v_hat(k) = vt(k) / (1.0_dp - beta2**t)
          denom(k) = sqrt(v_hat(k)) + epsilon
          w_xh(:, k) = w_xh(:, k) - alpha * m_hat(k) / denom(k)
      end do
      
      do k = 4, 6
          mt(k) = beta1 * mt(k) + (1.0_dp - beta1) * dw(k-3)
          vt(k) = beta2 * vt(k) + (1.0_dp - beta2) * (dw(k-3)**2)
          m_hat(k) = mt(k) / (1.0_dp - beta1**t)
          v_hat(k) = vt(k) / (1.0_dp - beta2**t)
          denom(k) = sqrt(v_hat(k)) + epsilon
          w_hh(:, k-3) = w_hh(:, k-3) - alpha * m_hat(k) / denom(k)
      end do
      
      do k = 7, 9
          mt(k) = beta1 * mt(k) + (1.0_dp - beta1) * dh(k-6)
          vt(k) = beta2 * vt(k) + (1.0_dp - beta2) * (dh(k-6)**2)
          m_hat(k) = mt(k) / (1.0_dp - beta1**t)
          v_hat(k) = vt(k) / (1.0_dp - beta2**t)
          denom(k) = sqrt(v_hat(k)) + epsilon
          w_hy(k-6) = w_hy(k-6) - alpha * m_hat(k) / denom(k)
      end do
      
      bh = bh - alpha * db(1)
      by = by - alpha * db(1)
  end do

  ! Predict the value based on the last three elements of the sequence
  h = [0.0_dp, 0.0_dp, 0.0_dp]
  do head = 1, num_heads
      h = h + tanh(matmul(w_xh, [sequence(seq_len-2), sequence(seq_len-1), sequence(seq_len)]) + matmul(w_hh, h) + bh) * attention_weights(seq_len-2:seq_len, seq_len, head)
  end do
  h = h / real(num_heads)
  y_pred = dot_product(w_hy, h) + by

  ! Display the prediction result
  write(*,'(a,F0.15)') 'Predicted value = ', y_pred

  ! Deallocate memory
  deallocate(sequence, attention_weights)

end program rnn_mh_att_3
