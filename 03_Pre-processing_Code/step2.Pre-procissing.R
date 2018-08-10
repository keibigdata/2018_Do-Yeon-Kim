rm(list=ls())
memory.size()

#package check & install & load
install.packages(c("pROC","gmodels","klaR","e1071"))
library(dplyr)
library(stringi)
library(tm)
library(pROC)
library(slam)
library(gmodels)
library(e1071)
library(klaR)
library(readr)
library(caret)
library(ggplot2)

# 이모지전처리 완료된 문서 가져오기
parsedData =read_csv("/Users/kei/Desktop/emoji_total_v1.csv")
View(parsedData)

#컬럼명 변경하기
colnames(parsedData) = c("id","pContent")
# 단어간 스페이스 하나 더 추가하기
parsedDataRe = parsedData
parsedDataRe$pContent = gsub(" ","  ",parsedDataRe$pContent)

##################################################################
#Text Pre-processing
##################################################################
#Corpus 생성
corp = VCorpus(VectorSource(parsedDataRe$pContent))
#특수문자 제거
corp = tm_map(corp, removePunctuation) # ^^, >< : 처리?
#텍스트문서 형식으로 변환
corp = tm_map(corp, PlainTextDocument)
#Sparse Terms 삭제
dtm <- removeSparseTerms(dtm, as.numeric(0.9995))
dtm
#Document Term Matrix 생성 (단어 Length는 2로 세팅)
dtm = DocumentTermMatrix(corp, control=list(removeNumbers=FALSE, wordLengths=c(1,Inf)))
dtm
#Covert to Dataframe
dtmDf = as.data.frame(as.matrix(dtm))
#중복 Column 삭제
dtmDf = dtmDf[,!duplicated(colnames(dtmDf))]
##dtmDf$id = parsedData$id
dtmDf$target = target_val$senticlass

#Traing Set, Test Set 만들기
sample_idx <- sample(1:nrow(dtmDf), size = nrow(dtmDf)*0.6)
length(sample_idx)
trainingSet = dtmDf[sample_idx,]
testSet = dtmDf[-sample_idx,]

trainingSet$target = as.factor(trainingSet$target)
