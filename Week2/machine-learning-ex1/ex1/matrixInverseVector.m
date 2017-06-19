function A_inv_b = matrixInverseVector(A, b, x_init, alpha)
  
  cost = (A*x_init) - b;
  x_new = x_init;
  
  while (norm(cost) ^ 2) < (10 ^ 5),
  
  grad_f = 2*A*((A*x_init) - b);
  x_new = x_init - (alpha * grad_f);
  x_init = x_new;
  cost = (A*x_init) - b;
  
  end
  
  
  A_inv_b = x_new;

end