! BSD 3-Clause No Military License
! Copyright © 2024, Piotr Bajdek. All Rights Reserved.

program rnn_add_att_3
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: sp = selected_real_kind(6, 37)
  real(sp), dimension(:), allocatable :: sequence
  real(sp), dimension(:,:), allocatable :: attention_weights, attention_scores
  real(sp) :: b = 0.0_sp, alpha = 0.001_sp, beta1 = 0.9_sp, beta2 = 0.999_sp, t = 0.0_sp
  real(sp) :: mt(9) = 0.0_sp, vt(9) = 0.0_sp, m_hat(9), v_hat(9), denom(9)
  real(sp), dimension(3) :: dw, db, dh, w_hy, wa, ua, va
  real(sp) :: y_pred, error, epsilon = 1.0E-8_sp, by, bh
  real(sp), dimension(3) :: h
  real(sp), dimension(3,3) :: w_xh, w_hh
  integer :: i, j, k, iterations = 100000, seq_len

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
  allocate(sequence(seq_len), attention_weights(seq_len, seq_len), attention_scores(seq_len, seq_len))
  rewind(10)
  do i=1, seq_len
      read(10,*) sequence(i)
  end do
  close(10)

  ! Initialize the weight vectors and biases
  w_xh = reshape([0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp], [3,3])
  w_hh = reshape([0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp, 0.0_sp], [3,3])
  w_hy = [0.0_sp, 0.0_sp, 0.0_sp]
  wa = [0.1_sp, 0.1_sp, 0.1_sp]
  ua = [0.1_sp, 0.1_sp, 0.1_sp]
  va = [0.1_sp, 0.1_sp, 0.1_sp]
  bh = 0.0_sp
  by = 0.0_sp
  h = [0.0_sp, 0.0_sp, 0.0_sp]

  ! Initialize attention weights and scores
  attention_weights = 0.0_sp
  attention_scores = 0.0_sp

  ! Optimization loop using Adam optimizer
  do i=1, iterations
      ! Initialize gradients
      dw = [0.0_sp, 0.0_sp, 0.0_sp]
      dh = [0.0_sp, 0.0_sp, 0.0_sp]
      db = [0.0_sp, 0.0_sp, 0.0_sp]

      ! Additive attention calculation
      do j = 4, seq_len
          do k = 1, seq_len
              attention_scores(j, k) = sum(wa * tanh(ua * sequence(j) + va * sequence(k)))
          end do
      end do

      ! Compute attention weights using softmax
      do j = 4, seq_len
          attention_weights(j, :) = exp(attention_scores(j, :)) / sum(exp(attention_scores(j, :)))
      end do

      ! Compute for each element of the sequence excluding the first three
      do j=4, seq_len
          ! Compute hidden state
          h = tanh(matmul(w_xh, [sequence(j-1), sequence(j-2), sequence(j-3)]) + matmul(w_hh, h) + bh)

          ! Compute output with attention
          y_pred = dot_product(w_hy, h * attention_weights(j, :)) + by

          error = y_pred - sequence(j)
          dw = dw + error * [sequence(j-1) * attention_weights(j-1, j), &
                             sequence(j-2) * attention_weights(j-2, j), &
                             sequence(j-3) * attention_weights(j-3, j)]
          dh = dh + error * h * attention_weights(j, :)
          db = db + error
      end do

      ! Normalize gradients
      dw = dw / (seq_len - 3)
      dh = dh / (seq_len - 3)
      db = db / (seq_len - 3)

      ! Update weights using the Adam optimization algorithm
      t = t + 1.0_sp
      do j = 1, 3
          mt(j) = beta1 * mt(j) + (1.0_sp - beta1) * dw(j)
          vt(j) = beta2 * vt(j) + (1.0_sp - beta2) * (dw(j)**2)
          m_hat(j) = mt(j) / (1.0_sp - beta1**t)
          v_hat(j) = vt(j) / (1.0_sp - beta2**t)
          denom(j) = sqrt(v_hat(j)) + epsilon
          w_xh(:, j) = w_xh(:, j) - alpha * m_hat(j) / denom(j)
      end do
      
      do j = 4, 6
          mt(j) = beta1 * mt(j) + (1.0_sp - beta1) * dw(j-3)
          vt(j) = beta2 * vt(j) + (1.0_sp - beta2) * (dw(j-3)**2)
          m_hat(j) = mt(j) / (1.0_sp - beta1**t)
          v_hat(j) = vt(j) / (1.0_sp - beta2**t)
          denom(j) = sqrt(v_hat(j)) + epsilon
          w_hh(:, j-3) = w_hh(:, j-3) - alpha * m_hat(j) / denom(j)
      end do
      
      do j = 7, 9
          mt(j) = beta1 * mt(j) + (1.0_sp - beta1) * dh(j-6)
          vt(j) = beta2 * vt(j) + (1.0_sp - beta2) * (dh(j-6)**2)
          m_hat(j) = mt(j) / (1.0_sp - beta1**t)
          v_hat(j) = vt(j) / (1.0_sp - beta2**t)
          denom(j) = sqrt(v_hat(j)) + epsilon
          w_hy(j-6) = w_hy(j-6) - alpha * m_hat(j) / denom(j)
      end do
      
      bh = bh - alpha * db(1)
      by = by - alpha * db(1)
  end do

  ! Predict the value based on the last three elements of the sequence
  h = tanh(matmul(w_xh, [sequence(seq_len-2), sequence(seq_len-1), sequence(seq_len)]) + matmul(w_hh, h) + bh)
  y_pred = dot_product(w_hy, h * attention_weights(seq_len, :)) + by

  ! Display the prediction result
  write(*,'(a,F0.6)') 'Predicted value = ', y_pred

  ! Deallocate memory
  deallocate(sequence, attention_weights, attention_scores)

end program rnn_add_att_3
