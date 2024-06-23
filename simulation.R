library(FRB)
library(vars)
library(mlVAR)
source('functionsRvar.R')
source('functionsIRF.R')


###simulation setting for bivariate case

T=500
h=10
K=2
lags=1
cor=0.8
sd=1
pars=matrix(c(0.6,0.3,0.3,0.6),2,2)
sigma.true=matrix(sd*c(1,cor,cor,1),2,2)
init=matrix(c(0,0),1,2)
lv=0.01


###run M simulations
M=100
result.irf=matrix(NA,M,4*3)
result.coef=matrix(NA,M,4*3)
result.fevd=matrix(NA,M,2*3)
result.ltm=matrix(NA,M,2*3)
result.sig=matrix(NA,M,4*3)


for(k in 1:M){
###generate data
set.seed(k+10)

data.whole=simulateVAR(pars,  means = 0, lags = lags, Nt = T, init, residuals = sigma.true, T/2)
data.ori=as.matrix(data.whole[1:T,])
cont=matrix(0,T,K)
cont.id=cbind(sample(ceiling(T*0.99),ceiling(T*lv)),sample(K,ceiling(T*lv),replace=TRUE))
cont[cont.id]=rnorm(nrow(cont.id), 10, 1)
data=data.ori+cont

result0=VAR(data,p=lags, type = "const")
result1=var.MLTS(data, reweight=TRUE, lags=lags, gamma=0.25, delta=0.05)
result2=var.MM(data, lags=lags, R=0)

resulttrue=result1
resulttrue$beta[,]=rbind(c(0,0),pars)
resulttrue$sigma[,]=sigma.true


result.coef0=as.vector(t(Bcoef(result0))[-3,])
result.coef1=as.vector(result1$beta[-1,])
result.coef2=as.vector(result2$beta[-1,])

result.coef[k,]=c(result.coef0, result.coef1, result.coef2)


irftrue=irf.Rvar(resulttrue,data,n.ahead = 10,boot = FALSE, ci = 0.95, runs = 100)
irf0=irf(result0,n.ahead = 10,boot = FALSE, ci = 0.95, runs = 100)
irf1=irf.Rvar(result1,data,n.ahead = 10,boot = FALSE, ci = 0.95, runs = 100)
irf2=irf.Rvar(result2,data,n.ahead = 10,boot = FALSE, ci = 0.95, runs = 100)


result.irf0=apply(cbind(irf0$irf$V1,irf0$irf$V2)-cbind(irftrue$irf$V1,irftrue$irf$V2), 2, rmse)
result.irf1=apply(cbind(irf1$irf$V1,irf1$irf$V2)-cbind(irftrue$irf$V1,irftrue$irf$V2), 2, rmse)
result.irf2=apply(cbind(irf2$irf$V1,irf2$irf$V2)-cbind(irftrue$irf$V1,irftrue$irf$V2), 2, rmse)

result.irf[k,]=c(result.irf0, result.irf1, result.irf2)

fevdtrue=fevd.Rvar(resulttrue,data,n.ahead = h)
fevd0=fevd(result0,n.ahead = h)
fevd1=fevd.Rvar(result1,data,n.ahead = h)
fevd2=fevd.Rvar(result2,data,n.ahead = h)

result.fevd0=apply(cbind(fevd0$V1[,1],fevd0$V2[,2])-cbind(fevdtrue$V1[,1],fevdtrue$V2[,2]), 2, rmse)
result.fevd1=apply(cbind(fevd1$V1[,1],fevd1$V2[,2])-cbind(fevdtrue$V1[,1],fevdtrue$V2[,2]), 2, rmse)
result.fevd2=apply(cbind(fevd2$V1[,1],fevd2$V2[,2])-cbind(fevdtrue$V1[,1],fevdtrue$V2[,2]), 2, rmse)

result.fevd[k,]=c(result.fevd0,result.fevd1,result.fevd2)


ltm0=solve(diag(K)-t(Bcoef(result0))[-3,])%*%t(Bcoef(result0))[3,]
ltm1=solve(diag(K)-result1$beta[-1,])%*%result1$beta[1,]
ltm2=solve(diag(K)-result2$beta[-1,])%*%result2$beta[1,]

result.ltm[k,]=c(ltm0,ltm1,ltm2)


res=residuals(result0)
sigma0=t(res)%*%res/(nrow(res)-ncol(res))

result.sig0=as.vector(sigma0)
result.sig1=as.vector(result1$sigma)
result.sig2=as.vector(result2$sigma)

result.sig[k,]=c(result.sig0, result.sig1, result.sig2)

}

coefave=matrix(apply(result.coef,2,mean),4,3)
irfave=matrix(apply(result.irf,2,mean),4,3)
fevdave=matrix(apply(result.fevd,2,mean),2,3)
ltmave=matrix(apply(result.ltm,2,mean),2,3)
sigave=matrix(apply(result.sig,2,mean),4,3)


saveRDS(list(coef=result.coef, irf=result.irf, fevd=result.fevd, ltm=result.ltm, sig=result.sig),
        file=paste("simulation","T",T,"cor",cor,"out",lv*100, sep=""))
