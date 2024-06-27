! BSD 3-Clause No Military License
! Copyright © 2024, Piotr Bajdek. All Rights Reserved.

program linear_3_t_test
  implicit none
  character(len=256) :: arg_1
  integer, parameter :: qp = selected_real_kind(33, 4931)
  real(qp), dimension(:), allocatable :: sequence
  real(qp) :: b = 0.0_qp, alpha = 0.001_qp, beta1 = 0.9_qp, beta2 = 0.999_qp, t = 0.0_qp
  real(qp) :: mt(3) = 0.0_qp, vt(3) = 0.0_qp, m_hat(3), v_hat(3), denom(3)
  real(qp), dimension(3) :: w, dw
  real(qp) :: db, y_pred, error
  real(qp) :: epsilon = 1.0E-8_qp
  integer :: i, j, iterations = 100000, seq_len

  ! Variables for statistical significance test
  real(qp) :: mse, se_pred, t_statistic, p_value
  real(qp), dimension(50) :: confidence_levels = [0.99_qp,  0.98_qp,  0.97_qp,  0.96_qp, 0.95_qp, 0.94_qp,  0.93_qp,  0.92_qp,  0.91_qp,  0.90_qp, 0.89_qp,  0.88_qp,  0.87_qp,  0.86_qp,  0.85_qp, 0.84_qp,  0.83_qp,  0.82_qp,  0.81_qp,  0.80_qp, 0.79_qp,  0.78_qp,  0.77_qp,  0.76_qp,  0.75_qp, 0.74_qp,  0.73_qp,  0.72_qp,  0.71_qp,  0.70_qp, 0.69_qp,  0.68_qp,  0.67_qp,  0.66_qp,  0.65_qp, 0.64_qp,  0.63_qp,  0.62_qp,  0.61_qp,  0.60_qp, 0.59_qp,  0.58_qp,  0.57_qp,  0.56_qp,  0.55_qp, 0.54_qp,  0.53_qp,  0.52_qp,  0.51_qp,  0.50_qp]
  real(qp), dimension(50) :: t_critical

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

  ! Optimization loop using Adam optimizer
  do i=1, iterations
      ! Initialize gradients
      dw = [0.0_qp, 0.0_qp, 0.0_qp]
      db = 0.0_qp

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

  ! Predict the value based on the last three elements of the sequence
  y_pred = w(1) * sequence(seq_len-2) &
      + w(2) * sequence(seq_len-3) &
      + w(3) * sequence(seq_len-4) &
      + b

  ! Calculate Mean Squared Error (MSE)
  mse = 0.0_qp
  do j=4, seq_len
      error = (w(1) * sequence(j-2) + w(2) * sequence(j-3) + w(3) * sequence(j-4) + b) - sequence(j)
      mse = mse + error**2
  end do
  mse = mse / (seq_len - 3)

  ! Calculate Standard Error of Prediction
  se_pred = sqrt(mse * (1.0_qp + 3.0_qp / (seq_len - 3)))  ! 3 is the number of predictors

  ! Calculate t-statistic
  t_statistic = (y_pred - sequence(seq_len)) / se_pred

  ! Calculate p-value using normal approximation
  p_value = 2.0_qp * (1.0_qp - normal_cdf(abs(t_statistic)))

  ! Calculate critical values for different confidence levels using normal approximation
  do i = 1, 50
    t_critical(i) = abs(normal_cdf_inverse((1.0_qp + confidence_levels(i)) / 2.0_qp))
  end do

  ! Display the prediction result and statistical significance
  write(*,'(a,F0.33)') 'Predicted value = ', y_pred
  write(*,'(a,F0.33)') 'Last actual value = ', sequence(seq_len)
  write(*,'(a,F0.33)') 'T-statistic = ', t_statistic
  write(*,'(a,F0.33)') 'P-value = ', p_value

  do i = 1, 50
    if (abs(t_statistic) > t_critical(i)) then
      if (sequence(seq_len) > y_pred) then
        write(*,'(a,F5.2,a)') 'At ', confidence_levels(i)*100, '% confidence level, the trend is likely increasing.'
      else
        write(*,'(a,F5.2,a)') 'At ', confidence_levels(i)*100, '% confidence level, the trend is likely decreasing.'
      end if
      exit
    end if
  end do

  if (abs(t_statistic) <= t_critical(50)) then
    write(*,'(a)') 'There is not enough evidence to determine a clear trend at any of the specified confidence levels.'
  end if

  ! Deallocate memory
  deallocate(sequence)

contains

  function normal_cdf(x) result(cdf)
    real(qp), intent(in) :: x
    real(qp) :: cdf
    real(qp), parameter :: a1 = 0.254829592_qp
    real(qp), parameter :: a2 = -0.284496736_qp
    real(qp), parameter :: a3 = 1.421413741_qp
    real(qp), parameter :: a4 = -1.453152027_qp
    real(qp), parameter :: a5 = 1.061405429_qp
    real(qp), parameter :: p = 0.3275911_qp
    real(qp) :: t, erf

    t = 1.0_qp / (1.0_qp + p * abs(x))
    erf = 1.0_qp - ((((a5*t + a4)*t + a3)*t + a2)*t + a1) * t * exp(-x**2)
    cdf = 0.5_qp * (1.0_qp + sign(1.0_qp, x) * erf)
  end function normal_cdf

  function normal_cdf_inverse(p) result(x)
    real(qp), intent(in) :: p
    real(qp) :: x
    real(qp) :: t, c0, c1, c2, d1, d2, d3

    t = sqrt(-2.0_qp * log(min(p, 1.0_qp - p)))
    c0 = 2.515517_qp
    c1 = 0.802853_qp
    c2 = 0.010328_qp
    d1 = 1.432788_qp
    d2 = 0.189269_qp
    d3 = 0.001308_qp

    x = t - (c0 + c1*t + c2*t**2) / (1.0_qp + d1*t + d2*t**2 + d3*t**3)

    if (p > 0.5_qp) x = -x
  end function normal_cdf_inverse

end program linear_3_t_test
