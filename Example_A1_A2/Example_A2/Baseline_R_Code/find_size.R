library(glmnet)
library(SuperLearner)
library(vimp)
library(xgboost)
library(MASS)
library(purrr)
iter = 500

#ExampleA1
gen_data1 = function(n,p1,p2, type, a, rho = 0.3){
  p = p1+p2
  X = mvrnorm(n, rep(0,p), toeplitz(rho^seq(0,p-1,1)))
  epsilon = rnorm(n,0,0.5) #homo 
  ##hetero---epsilon = rnorm(n,0,0.5)*exp(X[,2])/(1+exp(X[,2]))
  
  if(type==1){#linear model sparse H1
    beta = c(rep(1,2),rep(0,p1-2),rep(1,2)*a/sqrt(2),rep(0,p2-2))
    Y = X%*%beta+epsilon
  }
  if(type == 2){#dense H1
    beta = c(rep(1,2),rep(0,p1-2),rep(1,p2)*a/sqrt(p1))
    Y = X%*%beta+epsilon
  }
  Z = X[,1:p1]
  W = X[,(p1+1):p]
  return(list(Z,W,Y))
}

#Example A2
gen_data2 = function(n,p1,p2, type, a, rho = 0.3){
  p = p1+p2
  X = mvrnorm(n, rep(0,p), toeplitz(rho^seq(0,p-1,1)))
  epsilon = rnorm(n,0,0.5)
  if(type==1){#sparse H1
    Y = X[,1]+X[,2]+(a*(X[,p1+1]+X[,p1+2]+X[,p1+3]+X[,p1+4]+X[,p1+5]))^2+epsilon
  }
  if(type == 2){#dense H1
    Y = X[,1]+X[,2]+(a*rowSums(X[,(p1+1):(p1+floor(p2/2))]))^2+epsilon
  }
  Z = X[,1:p1]
  W = X[,(p1+1):p]
  return(list(Z,W,Y))
}

#Example A3
gen_data3 = function(n,p1,p2, type, a, rho = 0.3){
  p = p1+p2
  X = mvrnorm(n, rep(0,p), toeplitz(rho^seq(0,p-1,1)))
  epsilon = rnorm(n,0,0.5)
  if(type==1){# sparse H1
    Y = X[,1]-X[,2]+a*(exp(X[,p1+1])+exp(X[,p1+2]))+epsilon
    #sqrt(X[,1]^2+X[,2]^2)+a*(X%*%beta)^2+epsilon
  }
  if(type == 2){#dense H1
    Y = X[,1]-X[,2]+a*rowSums(exp(X[,(p1+1):(p1+p2)]))+epsilon
  }
  Z = X[,1:p1]
  W = X[,(p1+1):p]
  return(list(Z,W,Y))
}

#single-split test ---  return test statistics --- under H0 is chisq(1)
test1 = function(Z,W,Y,c){
  n = nrow(Z)
  p1 = ncol(Z)
  p2 = ncol(W)
  X = cbind(Z,W)
  p = p1+p2
  
  index = sample(1:n,ceiling(c*n),replace = F)
  Z_train = Z[index,];Z_test = Z[-index,]
  Y_train = Y[index];Y_test = Y[-index]
  X_train = X[index,];X_test = X[-index,]
  
  param <- list(max_depth = 3, eta = 0.1, min_child_weight = 1,
                subsample = 1,
                objective = "reg:squarederror", eval_metric = "rmse")
  dtrain <-xgb.DMatrix(Z_train, label = Y_train, nthread = 40)
  dtest <- xgb.DMatrix(Z_test, label =  Y_test, nthread = 40)
  watchlist <- list(train = dtrain, eval = dtest)
  XGB <- xgb.train(param, dtrain, nrounds = 100, watchlist,
                   early_stopping_rounds = 20, verbose = 0)
  
  h_train  = predict(XGB,Z_train)
  h_test  = predict(XGB,Z_test)
  y_tilde_train = Y_train - h_train
  y_tilde_test = Y_test - h_test
  param <- list(max_depth = 1, eta = 0.1,min_child_weight = 1,
                subsample = 1,
                objective = "reg:squarederror", eval_metric = "rmse")
  
  
  dtrain <-xgb.DMatrix(X_train, label = y_tilde_train, nthread = 40)
  dtest <- xgb.DMatrix(X_test, label =  y_tilde_test, nthread = 40)
  watchlist <- list(train = dtrain, eval = dtest)
  XGB2 <- xgb.train(param, dtrain, nrounds = 100, watchlist,
                    early_stopping_rounds = 20, verbose = 0)
  g_test = predict(XGB2,X_test)
  
  stat = sum((g_test- mean((Y_test-h_test)) )^2)/mean((Y_test-h_test)^2)
  stat_enhance = stat+sum(g_test^2)
  return(c(stat,stat_enhance))
}

####for single split   return c=n/N
choose_c = function(Z,W,Y){
  c_list = c(3/4,4/5,5/6,6/7,7/8,8/9,9/10)
  c_num = length(c_list)
  
  n = nrow(Z)
  p1 = ncol(Z)
  p2 = ncol(W)
  X = cbind(Z,W)
  p = p1+p2
  
  MM = 200
  temp = rep(0,MM)
  for(c in 1:c_num){
    cc = c_list[c]
    for(m in 1:MM){
      permu = sample(1:n,n,F)
      WW = W[permu,]
      temp[m] = test1(Z,WW,Y,cc)[1]###without power-enhance
      #temp[m] = test1(Z,WW,Y,cc)[2]###power-enhance
    }
    #cat(kk)
    if(mean(temp>qchisq(0.95,df = 1))<=0.05){
      break
    }
  }
  return(cc)
}

#meishausen 2009
mei_test1 = function(Z,W,Y,c,B=10){
  pv = matrix(0,2,B)
  #c = choose_c(Z,W,Y)
  for(b in 1:B){
    test_result = test1(Z,W,Y,c)
    pv[1,b] = 1-pchisq(test_result[1],df=1)
    pv[2,b] = 1-pchisq(test_result[2],df=1)
  }
  gamin = 0.05
  gammaArray = seq(gamin,1,0.01)
  Qgam=sapply(gammaArray,function(gam){
    min(quantile(pv[1,]/gam,gam,type=1),1)
  })
  Q1 = min(min(Qgam)*(1-log(gamin)),1)
  Qgam=sapply(gammaArray,function(gam){
    min(quantile(pv[2,]/gam,gam,type=1),1)
  })
  Q2 = min(min(Qgam)*(1-log(gamin)),1)
  return(c(Q1,Q2))
}

###Cauchy 2020
Cauchy_test1 = function(Z,W,Y,c,B=10){
  pv = matrix(0,2,B)
  Tx = matrix(0,2,B)
  #c = choose_c(Z,W,Y)
  for(b in 1:B){
    test_result = test1(Z,W,Y,c)
    pv[1,b] = 1-pchisq(test_result[1],df=1)
    Tx[1,b] = tan((1/2-pv[1,b])*pi)
    pv[2,b] = 1-pchisq(test_result[2],df=1)
    Tx[2,b] = tan((1/2-pv[2,b])*pi)
  }
  Tmean = mean(Tx[1,])
  Q1 = 1/2-atan(Tmean)/pi
  Tmean = mean(Tx[2,])
  Q2 = 1/2-atan(Tmean)/pi
  return(c(Q1,Q2))
}

gen_tuples1 = function(N,J=50){
  m = floor(N/log(N))
  ind_list = matrix(0, nrow = floor(N/m)*J, ncol = m)
  for(j in 1:J){
    pi = sample(1:N,N,F)
    ind_list[((j-1)*floor(N/m)+1):(j*floor(N/m)),] =
      matrix(pi[1:(floor(N/m)*m)],floor(N/m),m)
  }
  return(ind_list)
}

rank_test1 = function(Z,W,Y,c,L = 5, J = 50){
  N = nrow(Z)
  m = floor(N/log(N))
  B = J*floor(N/m)
  ind_list1 = gen_tuples1(N,J=J)
  H = matrix(0,B,L)
  for(b in 1:B){
    ind = ind_list1[b,]
    Z_ind = Z[ind,]; W_ind = W[ind,]; Y_ind = Y[ind]
    for(l in 1:L){
      H[b,l] = test1(Z_ind,W_ind,Y_ind,c)[1]###without power-enhance
    }
    #cat(b,'\r')
  }
  H_vector = c(H)
  HH = matrix(0,B,L)
  for(b in 1:B){
    for(l in 1:L){
      HH[b,l] = qchisq((rank(H_vector)[l+(b-1)*L]-0.5)/B/L,df=1)
    }
  }
  
  aggre = rep(0,L)
  for(l in 1:L){
    aggre[l] = test1(Z,W,Y,c)[1]
  }
  alpha=0.05
  return(mean(aggre)>quantile(rowMeans(HH),1-alpha))
}

####################################

###Lundborg
Lundborg = function(Z,W,Y,c=0.5){
  n = nrow(Z)
  p1 = ncol(Z)
  p2 = ncol(W)
  X = cbind(Z,W)
  p = p1+p2
  
  index = sample(1:n,ceiling(c*n),replace = F)
  Z_train = Z[index,];Z_test = Z[-index,]
  Y_train = Y[index];Y_test = Y[-index]
  X_train = X[index,];X_test = X[-index,]
  df_train = data.frame(Y_train,X_train)
  df_test = data.frame(Y_test,X_test)
  
  XGB = xgboost(data = X_train, Y_train,
                nround=50, max_depth = 3, verbose = 0,
                min_child_weight = 1, eta = 0.3, nthread = 4)
  g_train  = predict(XGB,X_train)
  g_test = predict(XGB,X_test)
  
  XGB2 = xgboost(data = Z_train, g_train,
                 nrounds=50, max_depth = 3, verbose = 0,
                 min_child_weight = 1, eta = 0.3, nthread = 4,
                 validate_parameters = TRUE)
  m_train  = predict(XGB2,Z_train)
  h_train  = g_train-m_train
  rho_hat = mean((Y_train-m_train)*h_train)
  
  XGB3 = xgboost(data = X_train, (Y_train-g_train)^2,
                 nrounds = 50, max_depth = 3, verbose = 0,
                 min_child_weight = 1, eta = 0.3, nthread = 4,
                 validate_parameters = TRUE)
  v_train = predict(XGB3,X_train)
  
  cc = 0
  while(T){
    if(mean((Y_train-g_train)^2/(max(0,v_train))+cc)>=1){
      break
    }else{
      cc = cc+0.01
    }
  }
  
  g_test  = predict(XGB,X_test)
  m_test  = predict(XGB2,Z_test)
  h_test  = (g_test-m_test)*sign(rho_hat)
  v_test = max(predict(XGB3,X_test),0)+cc
  f_test = h_test/v_test
  XGB4 = xgboost(data = Z_test, f_test,
                 nround=50, max_depth = 3,verbose = 0,
                 min_child_weight = 1, eta = 0.3, nthread = 4,
                 validate_parameters = TRUE)
  m_f_hat = predict(XGB4,Z_test)
  XGB5 = xgboost(data = Z_test, Y_test,
                 nround=50, max_depth = 3,verbose = 0,
                 min_child_weight = 1, eta = 0.3, nthread = 4,
                 validate_parameters = TRUE)
  m_hat = predict(XGB5,Z_test)
  L = (Y_test-m_hat)*(f_test-m_f_hat)
  nn = length(L)
  stat = sum(L)/sqrt(n)/(sqrt(mean(L^2)-mean(L)^2))
  return(stat)
}

Lundborg_M = function(Z,W,Y,c=0.5,B=6){
  return(mean(map_dbl(1:B,function(xxx,Z,W,Y,c){
    return(Lundborg(Z,W,Y,c))
  },Z,W,Y,c)))
}

Dai = function(Z,W,Y,c=0.5){
  n = nrow(Z)
  p1 = ncol(Z)
  p2 = ncol(W)
  X = cbind(Z,W)
  p = p1+p2
  
  index = sample(1:n,ceiling(c*n),replace = F)
  Z_train = Z[index,];Z_test = Z[-index,]
  Y_train = Y[index];Y_test = Y[-index]
  X_train = X[index,];X_test = X[-index,]
  df_train = data.frame(Y_train,X_train)
  df_test = data.frame(Y_test,X_test)
  
  XGB = xgboost(data = X_train, Y_train,
                nround=50, max_depth = 3, verbose = 0,
                min_child_weight = 1, eta = 0.3, nthread = 4)
  m_test = predict(XGB,X_test)
  
  XGB2 = xgboost(data = Z_train, Y_train,
                 nrounds=50, max_depth = 3, verbose = 0,
                 min_child_weight = 1, eta = 0.3, nthread = 4,
                 validate_parameters = TRUE)
  h_test  = predict(XGB2,Z_test)

  stat = (Y_test-m_test)^2-(Y_test-h_test)^2+rnorm(n)*0.01
  stat = sum(stat)/sqrt(n-ceiling(c*n))/sd(stat)
  return(stat)
}

Dai_M = function(Z,W,Y,c=0.5,B=10){
  pv = rep(0,B)
  Tx = rep(0,B)
  #c = choose_c(Z,W,Y)
  for(b in 1:B){
    test_result = Dai(Z,W,Y,c)
    pv[b] = pnorm(test_result)
    Tx[b] = tan((1/2-pv[b])*pi)
  }
  Tmean = mean(Tx)
  Q = 1/2-atan(Tmean)/pi
  return(Q)
}

func = function(xxx,n,p1,p2,c=0.8){
  p=p1+p2
  #Example A1
  data = gen_data1(n,p1,p2,1,0)#H0
  # data = gen_data1(n,p1,p2,1,0.5)#H1 sparse
  # data = gen_data1(n,p1,p2,2,1/sqrt(2*p2))#H1 dense
  # #Example A2
  # data = gen_data2(n,p1,p2,1,0)#H0
  # data = gen_data2(n,p1,p2,1,1/sqrt(2*5))#H1 sparse
  # data = gen_data2(n,p1,p2,2,1/sqrt(2*floor(p2/2)))#H1 dense
  # #Example A3
  # data = gen_data3(n,p1,p2,1,0)#H0
  # data = gen_data3(n,p1,p2,1,0.5*0.5)#H1 sparse
  # data = gen_data3(n,p1,p2,2,0.5*1/sqrt(2*p2))#H1 dense
  Z=data[[1]];W=data[[2]];Y=data[[3]]
  X=cbind(Z,W)
  #choose_c(Z,W,Y)
  return(c(
    test1(Z,W,Y,c),#test statistics
    Cauchy_test1(Z,W,Y,c),#pvalue
    Lundborg(Z,W,Y),#test statistic
    Lundborg_M(Z,W,Y),
    cv_vim(Y,data.frame(X),indx = seq(p1+1,p,1),SL.library = 'SL.xgboost',V=2)$p_value#,
    #Dai(Z,W,Y),
    #Dai_M(Z,W,Y)
  ))
}

cat('sample size:100: \n')
# for(n in (1:4)*100){
for(n in c(600, 800)){
  p1=25;p2=25
  rr = 123
  set.seed(rr)
  
  ans = matrix(0,iter,7)
  result = matrix(0,iter,7)
  for(i in 1:iter){
    ans[i,]=func(i,n,p1,p2,c=4/5)
    result[i,1] = mean(ans[1:i,1]>qchisq(0.95,df=1))
    result[i,2] = mean(ans[1:i,2]>qchisq(0.95,df=1))
    result[i,3] = mean(ans[1:i,3]<0.05)
    result[i,4] = mean(ans[1:i,4]<0.05)
    result[i,5] = mean(ans[1:i,5]>qnorm(0.95))
    result[i,6] = mean(ans[1:i,6]>qnorm(0.95))
    result[i,7] = mean(ans[1:i,7]<0.05)
    #result[i,8] = mean(ans[1:i,8]<qnorm(0.05))
    #result[i,9] = mean(ans[1:i,9]<0.05)
    if(i%%1==0){
      cat(i,' result: ',result[i,],'\r')
    }
  }
  result[iter,]
  #write.csv(ans,paste0('size_ans_n_300.csv'))
  write.csv(ans,paste0('size_ans_n_',n,'.csv'))
}

#write.csv(t(result[iter,]),paste0('result_n',n,'.csv'))
# [1] 0.078 0.104 0.140 0.214 0.006 0.000 0.064
