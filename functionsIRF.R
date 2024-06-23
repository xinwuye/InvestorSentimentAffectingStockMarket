###this R file contains functions used to compute IRF for each of the model.

##transfer coef into Acoef

Acoef.Rvar=function (x) 
{
  K <- dim(x$ydata)[2]
  p <- x$lags
  coef=cbind(t(x$beta)[,-1],t(x$beta)[,1])
  A <- coef[, 1:(K * p)]
  As <- list()
  start <- seq(1, p * K, K)
  end <- seq(K, p * K, K)
  for (i in 1:p) {
    As[[i]] <- matrix(A[, start[i]:end[i]], nrow = K, ncol = K)
    rownames(As[[i]]) <- rownames(A)
    colnames(As[[i]]) <- colnames(A[, start[i]:end[i]])
  }
  return(As)
}


###vars:::Psi.varest
Psi.Rvar=function (x, nstep = 10, ...) 
{
  nstep <- abs(as.integer(nstep))
  Phi <- Phi.Rvar(x, nstep = nstep)
  Psi <- array(0, dim = dim(Phi))
  sigma.u <- x$sigma
  P <- t(chol(sigma.u))
  dim3 <- dim(Phi)[3]
  for (i in 1:dim3) {
    Psi[, , i] <- Phi[, , i] %*% P
  }
  return(Psi)
}


### vars:::Phi.varest
Phi.Rvar=function (x, nstep = 10, ...) 
{
  nstep <- abs(as.integer(nstep))
  K <- dim(x$ydata)[2]
  p <- x$lags
  A <- as.array(Acoef.Rvar(x))
  if (nstep >= p) {
    As <- array(0, dim = c(K, K, nstep + 1))
    for (i in (p + 1):(nstep + 1)) {
      As[, , i] <- matrix(0, nrow = K, ncol = K)
    }
  }
  else {
    As <- array(0, dim = c(K, K, p))
  }
  for (i in 1:p) {
    As[, , i] <- A[[i]]
  }
  Phi <- array(0, dim = c(K, K, nstep + 1))
  Phi[, , 1] <- diag(K)
  Phi[, , 2] <- Phi[, , 1] %*% As[, , 1]
  if (nstep > 1) {
    for (i in 3:(nstep + 1)) {
      tmp1 <- Phi[, , 1] %*% As[, , i - 1]
      tmp2 <- matrix(0, nrow = K, ncol = K)
      idx <- (i - 2):1
      for (j in 1:(i - 2)) {
        tmp2 <- tmp2 + Phi[, , j + 1] %*% As[, , idx[j]]
      }
      Phi[, , i] <- tmp1 + tmp2
    }
  }
  return(Phi)
}

####vars:::.irf
.irf.Rvar=function (x, impulse, response, y.names, n.ahead, ortho, cumulative) 
{
    if (ortho) {
      irf <- Psi.Rvar(x, nstep = n.ahead)
    }
    else {
      irf <- Phi.Rvar(x, nstep = n.ahead)
    }
  dimnames(irf) <- list(y.names, y.names, NULL)
  idx <- length(impulse)
  irs <- list()
  for (i in 1:idx) {
    irs[[i]] <- matrix(t(irf[response, impulse[i], 1:(n.ahead + 
                                                        1)]), nrow = n.ahead + 1)
    colnames(irs[[i]]) <- response
    if (cumulative) {
      if (length(response) > 1) 
        irs[[i]] <- apply(irs[[i]], 2, cumsum)
      if (length(response) == 1) {
        tmp <- matrix(cumsum(irs[[i]]))
        colnames(tmp) <- response
        irs[[i]] <- tmp
      }
    }
  }
  names(irs) <- impulse
  result <- irs
  return(result)
}


### vars:::fevd.varest
fevd.Rvar=function (x, n.ahead = 10, ...) 
{

  n.ahead <- abs(as.integer(n.ahead))
  K <- dim(x$ydata)[2]
  p <- x$lags
  ynames <- colnames(x$ydata)
  msey <- .fecov.Rvar(x, n.ahead = n.ahead)
  Psi <- Psi.Rvar(x, nstep = n.ahead)
  mse <- matrix(NA, nrow = n.ahead, ncol = K)
  Omega <- array(0, dim = c(n.ahead, K, K))
  for (i in 1:n.ahead) {
    mse[i, ] <- diag(msey[, , i])
    temp <- matrix(0, K, K)
    for (l in 1:K) {
      for (m in 1:K) {
        for (j in 1:i) {
          temp[l, m] <- temp[l, m] + Psi[l, m, j]^2
        }
      }
    }
    temp <- temp/mse[i, ]
    for (j in 1:K) {
      Omega[i, , j] <- temp[j, ]
    }
  }
  result <- list()
  for (i in 1:K) {
    result[[i]] <- matrix(Omega[, , i], nrow = n.ahead, ncol = K)
    colnames(result[[i]]) <- ynames
  }
  names(result) <- ynames
  class(result) <- "varfevd"
  return(result)
}



###vars::: .fecov
.fecov.Rvar=function (x, n.ahead) 
{
  K <- dim(x$ydata)[2]
  p <- x$lags
  sigma.u <- x$sigma
  Sigma.yh <- array(NA, dim = c(K, K, n.ahead))
  Sigma.yh[, , 1] <- sigma.u
  Phi <- Phi.Rvar(x, nstep = n.ahead)
  if (n.ahead > 1) {
    for (i in 2:n.ahead) {
      temp <- matrix(0, nrow = K, ncol = K)
      for (j in 2:i) {
        temp <- temp + Phi[, , j] %*% sigma.u %*% t(Phi[, 
                                                        , j])
      }
      Sigma.yh[, , i] <- temp + Sigma.yh[, , 1]
    }
  }
  return(Sigma.yh)
}


###vars:::.boot
.boot.Rvar=function (x, data, n.ahead, runs, ortho, cumulative, impulse, response, 
          ci, seed, y.names) 
{
  if (!(is.null(seed))) 
    set.seed(abs(as.integer(seed)))
  
  K <- dim(x$ydata)[2]
  p <- x$lags
  
  obs <- dim(x$ydata)[1]
  total <- obs+p
  B <- t(x$beta)
  
  BOOT <- vector("list", runs)
  ysampled <- matrix(0, nrow = total, ncol = K)
  colnames(ysampled) <- colnames(x$ydata)

  resorig <- scale(x$res, scale = FALSE)

  for (i in 1:runs) {
    booted <- sample(c(1:obs), replace = TRUE)
    resid <- resorig[booted, ]
    lasty <- c(t(data[p:1, ]))
    ysampled[c(1:p), ] <- data[c(1:p), ]
    for (j in 1:obs) {
      lasty <- lasty[1:(K * p)]
      Z <- c(1, lasty)
      ysampled[j + p, ] <- B %*% Z + resid[j, ]
      lasty <- c(ysampled[j + p, ], lasty)
    }
    varboot <- update(x, data = ysampled)

    BOOT[[i]] <- .irf.Rvar(x = varboot, n.ahead = n.ahead, ortho = ortho, 
                      cumulative = cumulative, impulse = impulse, response = response, 
                      y.names = y.names)
  }
  lower <- ci/2
  upper <- 1 - ci/2
  mat.l <- matrix(NA, nrow = n.ahead + 1, ncol = length(response))
  mat.u <- matrix(NA, nrow = n.ahead + 1, ncol = length(response))
  Lower <- list()
  Upper <- list()
  idx1 <- length(impulse)
  idx2 <- length(response)
  idx3 <- n.ahead + 1
  temp <- rep(NA, runs)
  for (j in 1:idx1) {
    for (m in 1:idx2) {
      for (l in 1:idx3) {
        for (i in 1:runs) {
          if (idx2 > 1) {
            temp[i] <- BOOT[[i]][[j]][l, m]
          }
          else {
            temp[i] <- matrix(BOOT[[i]][[j]])[l, m]
          }
        }
        mat.l[l, m] <- quantile(temp, lower, na.rm = TRUE)
        mat.u[l, m] <- quantile(temp, upper, na.rm = TRUE)
      }
    }
    colnames(mat.l) <- response
    colnames(mat.u) <- response
    Lower[[j]] <- mat.l
    Upper[[j]] <- mat.u
  }
  names(Lower) <- impulse
  names(Upper) <- impulse
  result <- list(Lower = Lower, Upper = Upper)
  return(result)
}


###irf complete vars:::irf.varest
irf.Rvar=function (x, data, impulse = NULL, response = NULL, n.ahead = 10, ortho = TRUE, 
          cumulative = FALSE, boot = TRUE, ci = 0.95, runs = 100, seed = NULL, 
          ...) 
{
  
  y.names <- colnames(x$ydata)
  if (is.null(impulse)) {
    impulse <- y.names
  }
  else {
    impulse <- as.vector(as.character(impulse))
    if (any(!(impulse %in% y.names))) {
      stop("\nPlease provide variables names in impulse\nthat are in the set of endogenous variables.\n")
    }
    impulse <- subset(y.names, subset = y.names %in% impulse)
  }
  if (is.null(response)) {
    response <- y.names
  }
  else {
    response <- as.vector(as.character(response))
    if (any(!(response %in% y.names))) {
      stop("\nPlease provide variables names in response\nthat are in the set of endogenous variables.\n")
    }
    response <- subset(y.names, subset = y.names %in% response)
  }
  irs <- .irf.Rvar(x = x, impulse = impulse, response = response, 
              y.names = y.names, n.ahead = n.ahead, ortho = ortho, 
              cumulative = cumulative)
  Lower <- NULL
  Upper <- NULL
  if (boot) {
    ci <- as.numeric(ci)
    if ((ci <= 0) | (ci >= 1)) {
      stop("\nPlease provide a number between 0 and 1 for the confidence interval.\n")
    }
    ci <- 1 - ci
    BOOT <- .boot.Rvar(x = x, data=data, n.ahead = n.ahead, runs = runs, 
                  ortho = ortho, cumulative = cumulative, impulse = impulse, 
                  response = response, ci = ci, seed = seed, y.names = y.names)
    Lower <- BOOT$Lower
    Upper <- BOOT$Upper
  }
  result <- list(irf = irs, Lower = Lower, Upper = Upper, response = response, 
                 impulse = impulse, ortho = ortho, cumulative = cumulative, 
                 runs = runs, ci = ci, boot = boot)
  return(result)
}


exfevd=function (x,no){
  out=NULL
  for (i in 1:length(x)){
    extract=x[[i]][,no]
    out=cbind(out,extract)
  }
  colnames(out)=names(x)
  return(out)
}
