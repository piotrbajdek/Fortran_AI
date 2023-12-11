! BSD 3-Clause No Military License
! Copyright © 2023, Piotr Bajdek. All Rights Reserved.

program mish_att_8_8_8
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: qp = selected_real_kind(33, 4931)
  real(qp), dimension(:,:), allocatable :: attention_weights
  real(qp), dimension(:), allocatable :: sequence, hidden_state
  real(qp), dimension(8) :: w1, w2, w3, dw1, dw2, dw3
  real(qp) :: b1 = 0.0_qp, b2 = 0.0_qp, b3 = 0.0_qp
  real(qp) :: alpha = 0.001_qp, beta1 = 0.9_qp, beta2 = 0.999_qp, t = 0.0_qp
  real(qp) :: mt1(8), mt2(8), mt3(8), vt1(8), vt2(8), vt3(8)
  real(qp) :: m_hat1(8), m_hat2(8), m_hat3(8), v_hat1(8), v_hat2(8), v_hat3(8), denom1(8), denom2(8), denom3(8)
  real(qp) :: db1, db2, db3, y_pred, error
  real(qp) :: epsilon = 1.0E-8_qp
  integer :: i, j, iterations = 75, seq_len

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
  allocate(sequence(seq_len), hidden_state(seq_len))
  rewind(10)
  do i=1, seq_len
      read(10,*) sequence(i)
  end do
  close(10)

  ! Initialize the weight vectors
  w1 = [0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp]
  w2 = [0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp]
  w3 = [0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp]

  ! Initialize attention weights
  allocate(attention_weights(seq_len, seq_len))
  attention_weights = 0.0_qp

  ! Initialize hidden state
  hidden_state = 0.0_qp

  ! Optimization loop using Adam optimizer
  do i=1, iterations
      ! Initialize gradients
      dw1 = [0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp]
      dw2 = [0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp]
      dw3 = [0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp, 0.0_qp]

      ! Element-wise dot-product attention calculation
      do j = 1, seq_len
        attention_weights(:, j) = sequence(j) * sequence(:)
      end do

      ! Scale attention weights
      attention_weights = attention_weights / real(seq_len)

      ! Compute gradients using attention weights and update hidden state
      do j=9, seq_len
          hidden_state(j) = mish(dot_product(w1, sequence(j-8:j-1)) + b1) + &
                            mish(dot_product(w2, sequence(j-8:j-1)) + b2) + &
                            mish(dot_product(w3, sequence(j-8:j-1)) + b3)
          error = hidden_state(j) - sequence(j)

          ! Gradients for the first set of weights
          dw1 = dw1 + error * sequence(j-8:j-1) * attention_weights(j-8:j-1, j)
          db1 = db1 + error

          ! Gradients for the second set of weights
          dw2 = dw2 + error * sequence(j-8:j-1) * attention_weights(j-8:j-1, j)
          db2 = db2 + error

          ! Gradients for the third set of weights
          dw3 = dw3 + error * sequence(j-8:j-1) * attention_weights(j-8:j-1, j)
          db3 = db3 + error
      end do

      ! Normalize gradients
      dw1 = dw1 / (seq_len - 8)
      db1 = db1 / (seq_len - 8)

      dw2 = dw2 / (seq_len - 8)
      db2 = db2 / (seq_len - 8)

      dw3 = dw3 / (seq_len - 8)
      db3 = db3 / (seq_len - 8)

      ! Update weights using the Adam optimization algorithm for each set of weights
      t = t + 1.0_qp

      ! Update for the first set of weights
      mt1 = beta1 * mt1 + (1.0_qp - beta1) * dw1
      vt1 = beta2 * vt1 + (1.0_qp - beta2) * (dw1**2)
      m_hat1 = mt1 / (1.0_qp - beta1**t)
      v_hat1 = vt1 / (1.0_qp - beta2**t)
      denom1 = sqrt(v_hat1) + epsilon
      w1 = w1 - alpha * m_hat1 / denom1
      b1 = b1 - alpha * db1

      ! Update for the second set of weights
      mt2 = beta1 * mt2 + (1.0_qp - beta1) * dw2
      vt2 = beta2 * vt2 + (1.0_qp - beta2) * (dw2**2)
      m_hat2 = mt2 / (1.0_qp - beta1**t)
      v_hat2 = vt2 / (1.0_qp - beta2**t)
      denom2 = sqrt(v_hat2) + epsilon
      w2 = w2 - alpha * m_hat2 / denom2
      b2 = b2 - alpha * db2

      ! Update for the third set of weights
      mt3 = beta1 * mt3 + (1.0_qp - beta1) * dw3
      vt3 = beta2 * vt3 + (1.0_qp - beta2) * (dw3**2)
      m_hat3 = mt3 / (1.0_qp - beta1**t)
      v_hat3 = vt3 / (1.0_qp - beta2**t)
      denom3 = sqrt(v_hat3) + epsilon
      w3 = w3 - alpha * m_hat3 / denom3
      b3 = b3 - alpha * db3
  end do

  ! Predict the value based on the last eight elements of the sequence
  y_pred = mish(dot_product(w1, sequence(seq_len-8:seq_len-1)) + b1) + &
           mish(dot_product(w2, sequence(seq_len-8:seq_len-1)) + b2) + &
           mish(dot_product(w3, sequence(seq_len-8:seq_len-1)) + b3)

  ! Display the prediction result
  write(*,'(a,F0.33)') 'Predicted value = ', y_pred

  ! Deallocate memory
  deallocate(sequence, attention_weights, hidden_state)

contains

  real(qp) function mish(x)
    real(qp), intent(in) :: x
    mish = x * tanh(log(1 + exp(x)))
  end function mish

end program mish_att_8_8_8
