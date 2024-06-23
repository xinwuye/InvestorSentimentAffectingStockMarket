#####this file contains functions used to compute robust VAR models

##huber loss
rho.huber=function(x,cval){
  rho1=ifelse(abs(x)<=cval,(x^2/2),(cval*abs(x)-cval^2/2))
  return(rho1)}

##tukey loss
rho.tukey=function(x,cval){
  U1=cval^2/6*(1-(1-(x/cval)^2)^3)
  U1[abs(x)>cval]=(cval^2)/6
  return(U1)}

##function to perform lag selection for varMLTS using robust AIC BIC HQ

var.MLTS.select=function(data, reweight=TRUE, max.lag=5, gamma=0.25, delta=0.01, c=1.345){
  
  T=dim(data)[1]
  nvar=dim(data)[2]
  
  max.lag=max.lag
  rAIC=rep(NA,max.lag)
  rHQ=rep(NA,max.lag)
  rSC=rep(NA,max.lag)
  for (lags in 1:max.lag){
    
    ydata=data[(lags+1):T,]
    xdata=rep(1,T-lags)
    for (i in 1:lags){
      xdata = cbind(xdata,data[((1+lags)-i):(T-i),])
    }
    
    modelfit=mlts(xdata,ydata,gamma=gamma,ns=500,nc=10,delta=delta)
    
    if(reweight==TRUE){
      beta=modelfit$betaR
      sigma=modelfit$sigmaR
    } else{
      beta=modelfit$beta
      sigma=modelfit$sigma
    }
    res=ydata-xdata%*%beta
    phi = nvar*(lags*nvar+1)
    loglike=-(T-lags)/2*(log(2*pi)*nvar+log(det(sigma)))-sum(rho.huber(sqrt(diag(res%*%solve(sigma)%*%t(res))),c))
    #loglike=-(T-lags)/2*(log(2*pi)*nvar+log(det(sigma)))-sum(diag(res%*%solve(sigma)%*%t(res)))/2
    rAIC[lags]=-2/(T-lags)*(loglike-phi)
    rHQ[lags]=-2/(T-lags)*(loglike-log(log(T-lags))*phi)
    rSC[lags]=-2/(T-lags)*(loglike-log(T-lags)*phi/2)
  }
  
  AIC.lag=which.min(rAIC)
  HQ.lag=which.min(rHQ)
  SC.lag=which.min(rSC)
  
  criteria=cbind(rAIC,rHQ,rSC)
  
  return(list(criteria =criteria, AIC.lag=AIC.lag, HQ.lag=HQ.lag, SC.lag=SC.lag))
}


##function to perform lag selection for var.S using robust AIC BIC HQ

var.S.select=function(data, max.lag=5, R=0, bdp=0.25, conf =0.95, c=1.345){
  
  T=dim(data)[1]
  nvar=dim(data)[2]
  
  max.lag=max.lag
  rAIC=rep(NA,max.lag)
  rHQ=rep(NA,max.lag)
  rSC=rep(NA,max.lag)
  for (lags in 1:max.lag){
    
    ydata=data[(lags+1):T,]
    xdata=rep(1,T-lags)
    for (i in 1:lags){
      xdata = cbind(xdata,data[((1+lags)-i):(T-i),])
    }
    
    modelfit=FRBmultiregS(xdata, ydata, R=R, bdp = bdp, conf = conf)
    
    beta=modelfit$coefficients
    sigma=modelfit$Sigma
    
    res=ydata-xdata%*%beta
    phi = nvar*(lags*nvar+1)
    loglike=-(T-lags)/2*(log(2*pi)*nvar+log(det(sigma)))-sum(rho.huber(sqrt(diag(res%*%solve(sigma)%*%t(res))),c))
    #loglike=-(T-lags)/2*(log(2*pi)*nvar+log(det(sigma)))-sum(diag(res%*%solve(sigma)%*%t(res)))/2
    rAIC[lags]=-2/(T-lags)*(loglike-phi)
    rHQ[lags]=-2/(T-lags)*(loglike-log(log(T-lags))*phi)
    rSC[lags]=-2/(T-lags)*(loglike-log(T-lags)*phi/2)
  }
  
  AIC.lag=which.min(rAIC)
  HQ.lag=which.min(rHQ)
  SC.lag=which.min(rSC)
  
  criteria=cbind(rAIC,rHQ,rSC)
  
  return(list(criteria =criteria, AIC.lag=AIC.lag, HQ.lag=HQ.lag, SC.lag=SC.lag))
}


##function to perform lag selection for var.MM using robust AIC BIC HQ

var.MM.select=function(data, max.lag=5, R=0, conf =0.95, c=1.345){
  
  T=dim(data)[1]
  nvar=dim(data)[2]
  
  max.lag=max.lag
  rAIC=rep(NA,max.lag)
  rHQ=rep(NA,max.lag)
  rSC=rep(NA,max.lag)
  for (lags in 1:max.lag){
    
    ydata=data[(lags+1):T,]
    xdata=rep(1,T-lags)
    for (i in 1:lags){
      xdata = cbind(xdata,data[((1+lags)-i):(T-i),])
    }
    
    modelfit=FRBmultiregMM(xdata, ydata, R=R, conf = conf)
    
    beta=modelfit$coefficients
    sigma=modelfit$Sigma
    
    res=ydata-xdata%*%beta
    phi = nvar*(lags*nvar+1)
    loglike=-(T-lags)/2*(log(2*pi)*nvar+log(det(sigma)))-sum(rho.huber(sqrt(diag(res%*%solve(sigma)%*%t(res))),c))
    #loglike=-(T-lags)/2*(log(2*pi)*nvar+log(det(sigma)))-sum(diag(res%*%solve(sigma)%*%t(res)))/2
    rAIC[lags]=-2/(T-lags)*(loglike-phi)
    rHQ[lags]=-2/(T-lags)*(loglike-log(log(T-lags))*phi)
    rSC[lags]=-2/(T-lags)*(loglike-log(T-lags)*phi/2)
  }
  
  AIC.lag=which.min(rAIC)
  HQ.lag=which.min(rHQ)
  SC.lag=which.min(rSC)
  
  criteria=cbind(rAIC,rHQ,rSC)
  
  return(list(criteria =criteria, AIC.lag=AIC.lag, HQ.lag=HQ.lag, SC.lag=SC.lag))
}


##function to perform var.MLTS for a give lag

var.MLTS=function(data, reweight=TRUE, lags, gamma=0.25, delta=0.05){
  
  T=dim(data)[1]
  nvar=dim(data)[2]
  
  ydata=data[(lags+1):T,]
  xdata=rep(1,T-lags)
  for (i in 1:lags){
    xdata = cbind(xdata,data[((1+lags)-i):(T-i),])
  }
  
  modelfit=mlts(xdata,ydata,gamma=gamma,ns=500,nc=10,delta=delta)
  
  if(reweight==TRUE){
    beta=modelfit$betaR
    sigma=modelfit$sigmaR
    resD=modelfit$dresR
  }
  else{
    beta=modelfit$beta
    sigma=modelfit$sigma
    resD=modelfit$dres
  }
  res=ydata-xdata%*%beta
  call <- match.call()
  return(list(beta=beta,sigma=sigma,resD=resD,res=res,lags=lags, xdata=xdata, ydata=ydata, call=call))
}

##function to perform var.S for a give lag

var.S=function(data, lags, R=0, bdp=0.5, conf =0.95){
  
  T=dim(data)[1]
  nvar=dim(data)[2]
  
  ydata=data[(lags+1):T,]
  xdata=rep(1,T-lags)
  for (i in 1:lags){
    xdata = cbind(xdata,data[((1+lags)-i):(T-i),])
  }
  
  modelfit=FRBmultiregS(xdata, ydata, R=R, bdp = bdp, conf = conf)
  
  beta=modelfit$coefficients
  sigma=modelfit$Sigma
  CI.l=modelfit$CI.basic.lower
  CI.u=modelfit$CI.basic.upper
  
  res=ydata-xdata%*%beta
  resD=sqrt(diag(res%*%solve(sigma)%*%t(res)))
  call <- match.call()
  return(list(beta=beta,sigma=sigma,resD=resD,res=res,SE=modelfit$SE,cov=modelfit$cov,lags=lags, xdata=xdata, ydata=ydata, CI.l=CI.l, CI.u=CI.u, call=call))
}

##function to perform var.MM for a give lag

var.MM=function(data, lags, R=0, conf =0.95){
  
  T=dim(data)[1]
  nvar=dim(data)[2]
  
  ydata=data[(lags+1):T,]
  xdata=rep(1,T-lags)
  for (i in 1:lags){
    xdata = cbind(xdata,data[((1+lags)-i):(T-i),])
  }
  
  modelfit=FRBmultiregMM(xdata, ydata, R=R, conf = conf)
  
  beta=modelfit$coefficients
  sigma=modelfit$Sigma
  CI.l=modelfit$CI.basic.lower
  CI.u=modelfit$CI.basic.upper
  
  res=ydata-xdata%*%beta
  resD=sqrt(diag(res%*%solve(sigma)%*%t(res)))
  call <- match.call()
  return(list(beta=beta,sigma=sigma,resD=resD,res=res,SE=modelfit$SE,cov=modelfit$cov,lags=lags, xdata=xdata, ydata=ydata, CI.l=CI.l, CI.u=CI.u, call=call))
}



################################################################################
#mlts function Multivariate Regression function obtained from https://feb.kuleuven.be/public/u0017833/Programs/mlts/mlts.r.txt
mlts <- function(x,y,gamma,ns=500,nc=10,delta=0.01)
{ 
  d <- dim(x); n <- d[1]; p <- d[2]
  q <- ncol(y) 
  h <- floor(n*(1-gamma))+1
  obj0 <- 1e10 
  for (i in 1:ns)
  { sorted <- sort(runif(n),na.last = NA,index.return=TRUE)
  istart <- sorted$ix[1:(p+q)]
  xstart <- x[istart,]
  ystart <- y[istart,]
  bstart <- solve(t(xstart)%*%xstart,t(xstart)%*%ystart) 
  sigmastart <- (t(ystart-xstart%*%bstart))%*%(ystart-xstart%*%bstart)/q
  for (j in 1:nc)
  { res  <-  y - x %*% bstart
  tres <- t(res)
  dist2 <- colMeans(solve(sigmastart,tres)*tres)
  sdist2 <- sort(dist2,na.last = NA,index.return = TRUE)
  idist2 <- sdist2$ix[1:h]
  xstart <- x[idist2,]
  ystart <- y[idist2,]
  bstart <- solve(t(xstart)%*%xstart,t(xstart)%*%ystart)
  sigmastart <- (t(ystart-xstart%*%bstart))%*%(ystart-xstart%*%bstart)/(h-p)
  }
  obj <- det(sigmastart)
  if (obj < obj0)
  { result.beta <- bstart
  result.sigma <- sigmastart
  obj0 <- obj
  }
  }
  cgamma <- (1-gamma)/pchisq(qchisq(1-gamma,q),q+2)
  result.sigma <- cgamma * result.sigma
  res <- y - x %*% result.beta
  tres<-t(res)
  result.dres <- colSums(solve(result.sigma,tres)*tres)
  result.dres <- sqrt(result.dres)
  
  qdelta <- sqrt(qchisq(1-delta,q))
  good  <- (result.dres <= qdelta)
  xgood <- x[good,]
  ygood <- y[good,]
  result.betaR <- solve(t(xgood)%*%xgood,t(xgood)%*%ygood)
  result.sigmaR <- (t(ygood-xgood%*%result.betaR)) %*% 
    (ygood-xgood%*%result.betaR)/(sum(good)-p)
  cdelta <- (1-delta)/pchisq(qdelta^2,q+2)
  result.sigmaR<-cdelta*result.sigmaR
  resR<-y-x%*%result.betaR
  tresR<-t(resR)
  result.dresR <- colSums(solve(result.sigmaR,tresR)*tresR)
  result.dresR <- sqrt(result.dresR)
  list(beta=result.beta,sigma=result.sigma,dres=result.dres,
       betaR=result.betaR,sigmaR=result.sigmaR,dresR=result.dresR, good=good)
}

###h step ahead forecast  

forecastvar=function(model, h, ci) {
  lags=model$lags
  ydata=model$ydata
  xdata=model$xdata
  last=(ncol(xdata)-ncol(ydata)+1):ncol(xdata)
  forecast.y=matrix(nrow=h, ncol=ncol(ydata))
  forecast.x=matrix(nrow=h+1, ncol=ncol(xdata))
  forecast.x[1,]=c(1,ydata[nrow(ydata),],xdata[nrow(xdata),-c(1,last)])
  for(i in 1:h){
    forecast.y[i,]=forecast.x[i,]%*%model$beta
    forecast.x[i+1,]=c(1,forecast.y[i,],forecast.x[i,-c(1,last)])
  }
  colnames(forecast.y)=colnames(ydata)
  ynames=colnames(ydata)
  
  n.ahead=h
  K=ncol(ydata)
  yse <- matrix(NA, nrow = n.ahead, ncol = K)
  sig.y <- .fecov.Rvar(x = model, n.ahead = n.ahead)
  for (i in 1:n.ahead) {
    yse[i, ] <- sqrt(diag(sig.y[, , i]))
  }
  yse <- -1 * qnorm((1 - ci)/2) * yse
  colnames(yse) <- paste(ci, "of", ynames)
  colnames(forecast.y) <- paste(ynames, ".fcst", sep = "")
  lower <- forecast.y - yse
  colnames(lower) <- paste(ynames, ".lower", sep = "")
  upper <- forecast.y + yse
  colnames(upper) <- paste(ynames, ".upper", sep = "")
  forecasts <- list()
  for (i in 1:K) {
    forecasts[[i]] <- cbind(forecast.y[, i], lower[, i], upper[, 
                                                             i], yse[, i])
    colnames(forecasts[[i]]) <- c("fcst", "lower", 
                                  "upper", "CI")
  }
  names(forecasts) <- ynames

  return(list(fcst=forecasts))
}

###extract forecast only

forecast.pure=function(forc){
  forc.est=NULL
  for(i in 1:length(forc$fcst)){
    forc.est=cbind(forc.est, forc$fcst[[i]][,1])
  }
  colnames(forc.est)=names(forc$fcst)
  return(forc.est)
}


###RMSE

rmse=function(x){
  sqrt(sum(x^2)/length(x))
}

resD=function(x){
res=residuals(x)
sigma=t(res)%*%res/(nrow(res)-ncol(res))
resD=sqrt(diag(res%*%solve(sigma)%*%t(res)))
return(resD)
}


###long term mean
ltm.pure=function(x){
  K=x$K
  lags=x$p
  coef=t(Bcoef(x))[-ncol(Bcoef(x)),]
  intercept=t(Bcoef(x))[ncol(Bcoef(x)),]
  B=matrix(0,K,K)
  for(i in 1:lags){
  B=B+coef[((i-1)*K+1):(i*K),]
  }
  ltm=solve(diag(K)-B)%*%intercept
  return(ltm)
}

ltm.pure.Rvar=function(x){
  K=ncol(x$ydata)
  lags=x$lags
  coef=x$beta[-1,]
  intercept=x$beta[1,]
  B=matrix(0,K,K)
  for(i in 1:lags){
    B=B+coef[((i-1)*K+1):(i*K),]
  }
  ltm=solve(diag(K)-B)%*%intercept
  return(ltm)
}


###long term mean with boot ci

ltm=function(VAR, runs=100, ci=0.95){
  p <- VAR$p
  K <- VAR$K
  obs <- VAR$obs
  total <- VAR$totobs
  B <- Bcoef(VAR)
  BOOT <- matrix(NA, nrow=K, ncol=runs)
  ysampled <- matrix(0, nrow = total, ncol = K)
  colnames(ysampled) <- colnames(VAR$y)
  Zdet <- NULL
  if (ncol(VAR$datamat) > (K * (p + 1))) {
    Zdet <- as.matrix(VAR$datamat[, (K * (p + 1) + 1):ncol(VAR$datamat)])
  }
  resorig <- scale(resid(VAR), scale = FALSE)
  B <- Bcoef(VAR)
  for (i in 1:runs) {
    booted <- sample(c(1:obs), replace = TRUE)
    resid <- resorig[booted, ]
    lasty <- c(t(VAR$y[p:1, ]))
    ysampled[c(1:p), ] <- VAR$y[c(1:p), ]
    for (j in 1:obs) {
      lasty <- lasty[1:(K * p)]
      Z <- c(lasty, Zdet[j, ])
      ysampled[j + p, ] <- B %*% Z + resid[j, ]
      lasty <- c(ysampled[j + p, ], lasty)
    }
    varboot <- update(VAR, y = ysampled)
    BOOT[,i] <- ltm.pure(x = varboot)
  }
  ci=1-ci
  lower <- ci/2
  upper <- 1 - ci/2
  
  Lower <- apply(BOOT,1,quantile,prob=lower)
  Upper <- apply(BOOT,1,quantile,prob=upper)
  Mean=ltm.pure(VAR)
  ltm=cbind(Mean,Lower,Upper)
  
  return(ltm)
}

ltm.Rvar=function(x, runs=100, ci=0.95){

K <- dim(x$ydata)[2]
p <- x$lags

obs <- dim(x$ydata)[1]
total <- obs+p
B <- t(x$beta)

BOOT <- matrix(NA, nrow=K, ncol=runs)
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
  
  BOOT[,i] <- ltm.pure.Rvar(x = varboot)
}

ci=1-ci
lower <- ci/2
upper <- 1 - ci/2

Lower <- apply(BOOT,1,quantile,prob=lower)
Upper <- apply(BOOT,1,quantile,prob=upper)
Mean=ltm.pure.Rvar(x)
ltm=cbind(Mean,Lower,Upper)

return(ltm)

}